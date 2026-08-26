"""
app.py  —  Raspberry Pi 5 Dashboard Server
Flask + Flask-SocketIO bridge between browser dashboard and ESP32.

Run:  python app.py
Open: http://raspberrypi.local:5000   (or http://<pi-ip>:5000)

Install:
  pip install flask flask-socketio pyserial opencv-python numpy
"""
import shutil
import threading
import time
import serial
import serial.tools.list_ports
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
import os
import sys
import json
import subprocess
import cv2
from vision_detection import detect_red_object, compute_pixel_error,detect_green_object,detect_white_object,detect_yellow_object

#For tool calling purpose

# Optional YOLO support. Existing visual-servo behavior remains available
# even if Ultralytics/model weights are unavailable.
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ── Config ────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 115200
HOST         = "0.0.0.0"
PORT         = 3000



# ── YOLO incremental-training config ─────────────────────────────────────
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "./best.pt")
YOLO_CLASSES_FILE = os.environ.get("YOLO_CLASSES_FILE", "./classes.json")
TRAINING_ROOT = os.environ.get("YOLO_TRAINING_ROOT", "./training_corrections")
TRAINING_IMAGES = os.path.join(TRAINING_ROOT, "images")
TRAINING_LABELS = os.path.join(TRAINING_ROOT, "labels")
TRAINING_FRAMES = os.path.join(TRAINING_ROOT, "frames")
TRAINING_RUNS = os.path.join(TRAINING_ROOT, "runs")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))

# Visual servoing
DEAD_ZONE_X   = 70    # px
DEAD_ZONE_Y   = 70    # px
CMD_INTERVAL  = 0.08  # seconds between commands
CENTERED_HOLD = 0.1   # seconds to wait after CENTERED before re-evaluating
VS_TIMEOUT    = 15000.0  # seconds before auto-reset if grab never completes


# Servo calibration — mirrors all_servo[] in scripted/main.cpp
SERVO_CONFIG = [
    {"name": "Base",      "ch": 0, "max": 540, "min": 85,  "angle": 30,  "label": "Yaw"},
    {"name": "Shoulder",  "ch": 1, "max": 520, "min": 90,  "angle": 0, "label": "Pitch"},
    {"name": "Elbow",     "ch": 2, "max": 520, "min": 190, "angle": 180,  "label": "Pitch"},
    {"name": "Wrist",     "ch": 3, "max": 400, "min": 100, "angle": 150,  "label": "Pitch"},
    {"name": "WristRoll", "ch": 4, "max": 550, "min": 100, "angle": 90,  "label": "Roll"},
    {"name": "Gripper",   "ch": 5, "max": 400, "min": 80,  "angle": 0,   "label": "Open↔Close"},
]


# ── App ───────────────────────────────────────────────────────────────────
app    = Flask(__name__, static_folder=".", template_folder=".")
sio    = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


state = {
    "angles":    [c["angle"] for c in SERVO_CONFIG],
    # "distance":  0.0,
    "mode":      "manual",
    "arm_state": "IDLE",
    "detected":  False,
    "err_x":     0,
    "err_y":     0,
    "log":       [],
    "esp_connected": False,
}
state_lock   = threading.Lock()
ser_conn     = None
latest_frame = None
frame_lock   = threading.Lock()

# ── YOLO human-in-the-loop state ─────────────────────────────────────────
training_state = {"enabled": False, "images": 0, "corrections": 0, "boxes": 0,
                  "status": "idle", "message": "Training mode is OFF.", "training": False}
training_lock = threading.Lock()
latest_raw_frame = None
latest_yolo = []
yolo_model = None
yolo_lock = threading.Lock()

# Visual servoing runtime state
vs = {
    "active":        False,
    "pi_state":      "ALIGN",   # ALIGN | WAITING_DONE
    "last_cmd":      None,
    "last_cmd_time": 0.0,
    "centered_at":   0.0,
}
vs_lock = threading.Lock()

# Rolling ESP32 log for DONE/GRAB detection by VS logic
esp_log_lines = []



def ensure_training_dirs():
    for p in (TRAINING_ROOT, TRAINING_IMAGES, TRAINING_LABELS, TRAINING_FRAMES, TRAINING_RUNS):
        os.makedirs(p, exist_ok=True)

def load_yolo_classes():
    global yolo_model
    if yolo_model is not None:
        names = getattr(yolo_model, "names", None)
        if names:
            return {str(k): str(v) for k,v in names.items()} if isinstance(names,dict) else {str(i):str(v) for i,v in enumerate(names)}
    if os.path.exists(YOLO_CLASSES_FILE):
        try:
            with open(YOLO_CLASSES_FILE,"r",encoding="utf-8") as f:
                return {str(k):str(v) for k,v in json.load(f).items()}
        except Exception as e:
            add_log(f"YOLO classes file error: {e}","warn")
    return {}

def ensure_yolo_model():
    global yolo_model
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")
    if yolo_model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise RuntimeError(f"YOLO weights not found: {YOLO_MODEL_PATH}")
        add_log(f"Loading YOLO model: {YOLO_MODEL_PATH}","info")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        add_log("YOLO model loaded for Training Mode","ok")
    return yolo_model

def run_yolo_on_frame(frame):
    model = ensure_yolo_model()
    results = model.predict(frame, conf=YOLO_CONF, verbose=False)
    result = results[0]
    detections=[]
    if result.boxes is not None:
        for b in result.boxes:
            xyxy=b.xyxy[0].tolist()
            detections.append({"x1":float(xyxy[0]),"y1":float(xyxy[1]),
                               "x2":float(xyxy[2]),"y2":float(xyxy[3]),
                               "class_id":int(b.cls[0].item()),
                               "confidence":float(b.conf[0].item())})
    return detections

def set_training_status(status,message):
    with training_lock:
        training_state["status"]=status
        training_state["message"]=message
        training_state["training"]=(status=="training")
        data=dict(training_state)
    sio.emit("training_status",data)

def save_training_frame(frame,image_id):
    ensure_training_dirs()
    path=os.path.join(TRAINING_FRAMES,f"{image_id}.jpg")
    cv2.imwrite(path,frame,[cv2.IMWRITE_JPEG_QUALITY,92])
    return path

def write_yolo_label(path,detections,width,height):
    lines=[]
    for d in detections:
        cls=int(d["class_id"])
        x1=max(0,min(width,float(d["x1"]))); y1=max(0,min(height,float(d["y1"])))
        x2=max(0,min(width,float(d["x2"]))); y2=max(0,min(height,float(d["y2"])))
        bw=(x2-x1)/width; bh=(y2-y1)/height
        if bw<=0 or bh<=0: continue
        cx=((x1+x2)/2)/width; cy=((y1+y2)/2)/height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+("\n" if lines else ""))

def start_training_job(epochs=20):
    def worker():
        try:
            set_training_status("training","Incremental YOLO training is running in the background.")
            if YOLO is None: raise RuntimeError("Ultralytics is not installed.")
            if not os.path.exists(YOLO_MODEL_PATH): raise RuntimeError(f"YOLO weights not found: {YOLO_MODEL_PATH}")
            image_files=[f for f in os.listdir(TRAINING_IMAGES) if f.lower().endswith((".jpg",".jpeg",".png"))]
            if not image_files: raise RuntimeError("No saved correction images are available.")
            names=load_yolo_classes()
            if not names: raise RuntimeError("No class mapping found in model/classes.json.")
            yaml_path=os.path.join(TRAINING_ROOT,"correction_dataset.yaml")
            with open(yaml_path,"w",encoding="utf-8") as f:
                f.write(f"path: {os.path.abspath(TRAINING_ROOT)}\ntrain: images\nval: images\nnames:\n")
                for k in sorted(names,key=lambda x:int(x)): f.write(f"  {k}: {names[k]}\n")
            model=YOLO(YOLO_MODEL_PATH)
            run_name=time.strftime("incremental_%Y%m%d_%H%M%S")
            model.train(data=yaml_path,epochs=int(epochs),imgsz=640,project=TRAINING_RUNS,name=run_name,exist_ok=False,verbose=True)
            new_best=os.path.join(TRAINING_RUNS,run_name,"weights","best.pt")
            set_training_status("completed",f"Training completed. Candidate model: {new_best}")
            add_log(f"Candidate model created: {new_best}","ok")
        except Exception as e:
            set_training_status("error",f"Training failed: {e}")
            add_log(f"YOLO incremental training failed: {e}","error")
        finally:
            with training_lock: training_state["training"]=False
    threading.Thread(target=worker,daemon=True).start()


def decide_command(err_x, err_y, area):
    in_x = abs(err_x) <= DEAD_ZONE_X        #to avoid jittering, we define a dead zone for x and y axis. If the error is within the dead zone, we consider it centered.
    in_y = abs(err_y) <= DEAD_ZONE_Y        #to avoid jittering, we define a dead zone for x and y axis. If the error is within the dead zone, we consider it centered.
    in_area = area > 80000  # Example threshold for object size
    in_area1=area>20000 and area<=50000
    in_area2=area>50000 and area<=80000
    speed="4"


    if in_area1:
        speed="2";
    if in_area2:
        speed="1";

    if in_area:
        return "CATCH"
    if in_x and in_y:
        return "CENTERED:"+speed
    if not in_x:
        return "RIGHT:"+speed if err_x > 0 else "LEFT:"+speed
    if not in_y:
        return "FORWARD:"+speed if err_y > 0 else "BACK:"+speed
    return None


def add_log(msg, level="info"):
    entry = {"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    with state_lock:
        state["log"].append(entry)
        if len(state["log"]) > 500:
            state["log"].pop(0)
    sio.emit("log", entry)


def get_serial():
    """Try to open serial port, return connection or None."""
    global ser_conn
    try:
        ser_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(1.5)
        add_log(f"Serial connected: {SERIAL_PORT}", "ok")
        with state_lock:
            state["esp_connected"] = True
        sio.emit("status", {"esp": True})
        return ser_conn
    except serial.SerialException as e:
        add_log(f"Serial error: {e}", "error")
        with state_lock:
            state["esp_connected"] = False
        sio.emit("status", {"esp": False})
        return None


def serial_reader_thread():
    """Parse ESP32 serial output and push updates over SocketIO."""
    global ser_conn
    while True:
        if ser_conn is None or not ser_conn.is_open:
            ser_conn = get_serial()
            if ser_conn is None:
                time.sleep(3)
                continue

        try:
            if ser_conn.in_waiting:
                line = ser_conn.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                esp_log_lines.append(line)
                # Skip routine status spam from the firmware loop
                if any(line.startswith(p) for p in ("TargetServo:", "Target Angle:", "Current Angle:")):
                    continue
                add_log(f"[ESP] {line}", "muted")
                if len(esp_log_lines) > 50:
                    esp_log_lines.pop(0)

                # Parse distance
                # if line.startswith("Distance:"):
                #     try:
                #         dist = float(line.split(":")[1].strip())
                #         with state_lock:
                #             state["distance"] = round(dist, 1)
                #         sio.emit("sensor", {"distance": dist})
                #     except ValueError:
                #         pass

                # Parse state
                # if "state" in line.lower() or "ALIGN" in line or "GRAB" in line or "DONE" in line:
                #     for s in ["ALIGN", "APPROACH", "GRAB", "DONE"]:
                #         if s in line:
                #             with state_lock:
                #                 state["arm_state"] = s
                #             sio.emit("arm_state", {"state": s})
                #             break

        except (serial.SerialException, OSError) as e:
            add_log(f"Serial read error: {e}", "error")
            ser_conn = None
            with state_lock:
                state["esp_connected"] = False
            sio.emit("status", {"esp": False})

        time.sleep(0.005)


def send_serial(cmd: str):
    global ser_conn

    if ser_conn and ser_conn.is_open:
        try:
            ser_conn.write((cmd + "\n").encode())
            ser_conn.flush()
            add_log(f"→ {cmd}", "info")
            return True
        except serial.SerialException as e:
            add_log(f"Send error: {e}", "error")
    else:
        add_log("ESP32 not connected", "warn")
    return False

#To give the send_serial functionality for tool calling


# ── Camera thread ─────────────────────────────────────────────────────────
def camera_thread():
    global latest_frame
    cap = None

    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                add_log("Camera not available — retrying in 5s", "warn")
                cap = None
                time.sleep(5)
                continue
            add_log("Camera opened (index 0)", "ok")

        ret, frame = cap.read()
        if not ret:
            add_log("Camera read failed — reconnecting", "warn")
            cap.release()
            cap = None
            time.sleep(2)
            continue

        global latest_raw_frame, latest_yolo
        with frame_lock:
            latest_raw_frame = frame.copy()
        with training_lock:
            training_enabled = training_state["enabled"]
        if training_enabled:
            try:
                with yolo_lock:
                    latest_yolo = run_yolo_on_frame(frame)
            except Exception as e:
                with yolo_lock:
                    latest_yolo = []
                add_log(f"YOLO inference error: {e}","warn")

        # result = detect_red_object(frame)      #for coordinate of red
        result = detect_green_object(frame)      #for coordinate of red

       
        annotated = result["annotated"]              #for marking and design

        if training_enabled:
            try:
                names=load_yolo_classes()
                with yolo_lock: overlay=list(latest_yolo)
                for d in overlay:
                    x1,y1,x2,y2=map(int,[d["x1"],d["y1"],d["x2"],d["y2"]])
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,229,160),2)
                    label=f'{names.get(str(d["class_id"]),"unknown")} #{d["class_id"]} {d["confidence"]:.0%}'
                    cv2.putText(annotated,label,(x1,max(14,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,229,160),1,cv2.LINE_AA)
            except Exception:
                pass
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
        with frame_lock:
            latest_frame = buf.tobytes()

        err_x, err_y = 0, 0
        if result["found"] and result["centroid"]:
            err_x, err_y = compute_pixel_error(result["centroid"], frame.shape)
            area= float(result["area"])

        with state_lock:
            state["detected"] = result["found"]
            state["err_x"]    = err_x
            state["err_y"]    = err_y


        # Update UI for err x and y
        sio.emit("vision_update", {
            "found":    result["found"],
            "centroid": list(result["centroid"]) if result["centroid"] else None,
            "err_x":    err_x,
            "err_y":    err_y,
            "area":     float(result["area"]),
        })

        # ── Visual servoing logic ──────────────────────────────────────────
        # Decide under the lock, then send outside to avoid lock-order deadlock
        # (send_serial → add_log → state_lock must not run while vs_lock is held)
        cmd_to_send = None
        vs_log      = None
        vs_emit     = None

        with vs_lock:
            if vs["active"]:
                now = time.time()
                #CMD_INTERVAL is the time interval between commands. If the last command was sent more than CMD_INTERVAL seconds ago, we can send a new command.
                if result["found"] and (now - vs["last_cmd_time"]) >= CMD_INTERVAL:
                    if vs["pi_state"] == "ALIGN":
                        cmd = decide_command(err_x, err_y,area)
                        if cmd=="CATCH":
                            vs["pi_state"]      = "To_CATCH"
                            vs["last_cmd"]      = "CATCH"
                            vs["last_cmd_time"] = now
                            cmd_to_send = "CATCH"
                        elif cmd == "CENTERED":
                            # vs["pi_state"]      = "WAITING_DONE"
                            # vs["centered_at"]   = now
                            vs["last_cmd"]      = "CENTERED"
                            vs["last_cmd_time"] = now
                            cmd_to_send = "CENTERED"
                            vs_log = ("[VS] Centered — waiting for grab", "ok")
                        # elif cmd and cmd != vs["last_cmd"]:
                        #     vs["last_cmd"]      = cmd
                        #     vs["last_cmd_time"] = now
                        #     cmd_to_send = cmd

                        elif cmd:
                            vs["last_cmd"]      = cmd
                            vs["last_cmd_time"] = now
                            cmd_to_send = cmd

                    elif vs["pi_state"] == "WAITING_DONE":
                        for line in esp_log_lines[-10:]:
                            if "DONE" in line or "GRAB" in line:
                                vs["pi_state"] = "ALIGN"
                                vs["last_cmd"] = None
                                vs_log = ("[VS] Grab complete — reset to ALIGN", "ok")
                                break
                        if now - vs["centered_at"] > VS_TIMEOUT:
                            vs["pi_state"] = "ALIGN"
                            vs["last_cmd"] = None
                            vs_log = ("[VS] Timeout — reset to ALIGN", "warn")

                elif not result["found"] and vs["pi_state"] != "ALIGN":
                    vs["pi_state"] = "ALIGN"
                    vs["last_cmd"] = None

                elif vs["pi_state"]=="TO_CATCH":
                    for line in esp_log_lines[-10:]:
                        if "DONE" in line or "GRAB" in line:
                            vs["pi_state"] = "ALIGN"
                            vs["last_cmd"] = None
                            vs_log = ("[VS] Grab complete — reset to ALIGN", "ok")
                            break
                    if now - vs["last_cmd_time"] > VS_TIMEOUT:
                        vs["pi_state"] = "ALIGN"
                        vs["last_cmd"] = None
                        vs_log = ("[VS] Timeout — reset to ALIGN", "warn")

                vs_emit = {"pi_state": vs["pi_state"], "last_cmd": vs["last_cmd"]}

        if cmd_to_send:
            send_serial(cmd_to_send)
        if vs_log:
            add_log(*vs_log)
        if vs_emit:
            sio.emit("vs_update", vs_emit)

def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)  # ~30 fps cap


# ── REST API ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard_v3.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")




#For direct commands on the UI (LEFT/RIGHT/FORWARD/BACK/CENTERED).
@app.route("/api/command", methods=["POST"])
def api_command():
    """Send a raw command string to ESP32 (LEFT/RIGHT/FORWARD/BACK/CENTERED)."""
    data = request.get_json()
    cmd  = data.get("cmd", "").upper().strip()
    VALID = {"LEFT", "RIGHT", "FORWARD", "BACK", "CENTERED"}
    if cmd not in VALID:
        return jsonify({"ok": False, "error": f"Unknown command: {cmd}"}), 400
    ok = send_serial(cmd+":8")
    return jsonify({"ok": ok})

@app.route("/api/gripper/<action>", methods=["POST"])
def api_gripper(action):
    """Open or close gripper by sending matching serial command."""
    # Visual servo firmware responds to FORWARD/BACK/LEFT/RIGHT for alignment
    # Gripper is triggered internally by the state machine; we expose a manual override
    # by encoding as a comment — in manual mode the Pi can send direct angle commands
    if action == "open":
        add_log("Gripper OPEN (manual)", "warn")
        ok=send_serial("S5:0")
        return jsonify({"ok":ok})
    elif action == "close":
        add_log("Gripper CLOSE (manual)", "warn")
        ok=send_serial("S5:180")
        return jsonify({"ok": ok})
    return jsonify({"ok": False}), 400


@app.route("/api/mode", methods=["POST"])
def api_mode():
    data = request.get_json()
    mode = data.get("mode", "manual")
    with state_lock:
        state["mode"] = mode
    with vs_lock:
        vs["active"]   = (mode == "visual")
        vs["pi_state"] = "ALIGN"
        vs["last_cmd"] = None
    if mode == "manual" and ser_conn and ser_conn.is_open:
        try:
            ser_conn.reset_input_buffer()
            ser_conn.reset_output_buffer()
        except Exception:
            pass
    add_log(f"Mode → {mode.upper()}", "info")
    sio.emit("mode_change", {"mode": mode})
    sio.emit("vs_update", {"pi_state": "ALIGN", "last_cmd": None})
    return jsonify({"ok": True, "mode": mode})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the visual servoing state."""
    with vs_lock:
        vs["active"]   = False
        vs["pi_state"] = "ALIGN"
        vs["last_cmd"] = None
    with state_lock:
        state["mode"] = "manual"
    send_serial("RESET")
    add_log("Visual servoing reset to ALIGN", "info")
    sio.emit("mode_change", {"mode": "manual"})
    sio.emit("vs_update", {"pi_state": "ALIGN", "last_cmd": None})
    return jsonify({"ok": True})

INVERTED_SERVOS = {1, 2, 3}  # shoulder, elbow, wrist — physically mounted inverted


@app.route("/api/servo", methods=["POST"])
def api_servo():
    """Move a single servo — manual mode only. Body: {servo: 0-5, angle: 0-180}"""
    data  = request.get_json()
    idx   = data.get("servo")
    angle = data.get("angle")
    if idx is None or angle is None:
        return jsonify({"ok": False, "error": "Missing servo or angle"}), 400
    if not (0 <= int(idx) <= 5):
        return jsonify({"ok": False, "error": "Servo index out of range"}), 400
    if not (0 <= int(angle) <= 180):
        return jsonify({"ok": False, "error": "Angle out of range"}), 400
    with state_lock:
        if state["mode"] != "manual":
            return jsonify({"ok": False, "error": "Only available in manual mode"}), 403
    physical = 180 - int(angle) if int(idx) in INVERTED_SERVOS else int(angle)
    ok = send_serial(f"S{int(idx)}:{physical}")
    if ok:
        with state_lock:
            state["angles"][int(idx)] = int(angle)
        sio.emit("joint_update", {"angles": state["angles"]})
    return jsonify({"ok": ok})




# ── YOLO training / annotation API ───────────────────────────────────────
@app.route("/api/training/mode",methods=["POST"])
def api_training_mode():
    enabled=bool((request.get_json(silent=True) or {}).get("enabled",False))
    if enabled:
        try: ensure_yolo_model()
        except Exception as e: return jsonify({"ok":False,"error":str(e)}),400
    with training_lock: training_state["enabled"]=enabled
    set_training_status("ready" if enabled else "idle","Training mode is ON — YOLO validation is active." if enabled else "Training mode is OFF.")
    add_log("YOLO Training Mode → "+("ON" if enabled else "OFF"),"ok" if enabled else "info")
    return jsonify({"ok":True,"enabled":enabled})

@app.route("/api/training/classes")
def api_training_classes():
    try:
        ensure_yolo_model()
        return jsonify({"ok":True,"classes":load_yolo_classes()})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e),"classes":{}}),400

@app.route("/api/training/status")
def api_training_status():
    with training_lock: return jsonify(dict(training_state))

@app.route("/api/training/capture")
def api_training_capture():
    with training_lock:
        if not training_state["enabled"]: return jsonify({"ok":False,"error":"Enable Training Mode first."}),400
    with frame_lock: frame=None if latest_raw_frame is None else latest_raw_frame.copy()
    if frame is None: return jsonify({"ok":False,"error":"Camera frame is not available yet."}),503
    try: detections=run_yolo_on_frame(frame)
    except Exception as e: return jsonify({"ok":False,"error":str(e)}),400
    image_id=f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}"
    save_training_frame(frame,image_id)
    return jsonify({"ok":True,"image_id":image_id,"image_url":f"/api/training/frame/{image_id}",
                    "width":int(frame.shape[1]),"height":int(frame.shape[0]),"detections":detections})

@app.route("/api/training/frame/<image_id>")
def api_training_frame(image_id):
    filename=f"{image_id}.jpg"; path=os.path.join(TRAINING_FRAMES,filename)
    if not os.path.exists(path): return jsonify({"ok":False,"error":"Training frame not found."}),404
    return send_from_directory(TRAINING_FRAMES,filename,mimetype="image/jpeg")

@app.route("/api/training/save",methods=["POST"])
def api_training_save():
    data=request.get_json(silent=True) or {}
    image_id=str(data.get("image_id","")).strip(); detections=data.get("detections",[])
    width=int(data.get("width",0)); height=int(data.get("height",0))
    if not image_id or width<=0 or height<=0: return jsonify({"ok":False,"error":"Invalid training sample."}),400
    src=os.path.join(TRAINING_FRAMES,f"{image_id}.jpg")
    if not os.path.exists(src): return jsonify({"ok":False,"error":"Training frame not found."}),404
    ensure_training_dirs()
    shutil.copy2(src,os.path.join(TRAINING_IMAGES,f"{image_id}.jpg"))
    write_yolo_label(os.path.join(TRAINING_LABELS,f"{image_id}.txt"),detections,width,height)
    with training_lock:
        training_state["images"]+=1; training_state["boxes"]+=len(detections); training_state["corrections"]+=1
        out=dict(training_state)
    sio.emit("training_status",out)
    return jsonify({"ok":True,**out})

@app.route("/api/training/train",methods=["POST"])
def api_training_train():
    epochs=int((request.get_json(silent=True) or {}).get("epochs",20))
    with training_lock:
        if training_state["training"]: return jsonify({"ok":False,"error":"Training is already running."}),409
        if training_state["images"]==0: return jsonify({"ok":False,"error":"No correction samples saved yet."}),400
    start_training_job(max(1,min(200,epochs)))
    return jsonify({"ok":True,"status":"training","message":"Training started in background."})

@app.route("/api/ports")
def api_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports})




# ── SocketIO events ────────────────────────────────────────────────────────

@sio.on("connect")
def on_connect():
    with state_lock:
        s = dict(state)
    emit("init", s)
    add_log("Dashboard client connected", "ok")


@sio.on("update_angles")
def on_update_angles(data):
    """Browser pushes new angle values (for live display sync)."""
    angles = data.get("angles", [])
    if len(angles) == 6:
        with state_lock:
            state["angles"] = angles
        sio.emit("joint_update", {"angles": angles})




if __name__ == "__main__":
    ensure_training_dirs()
    print(f"[DASHBOARD] Starting on http://0.0.0.0:{PORT}")
    threading.Thread(target=serial_reader_thread, daemon=True).start()

    threading.Thread(target=camera_thread,daemon=True).start()
   
   
    # Always boot in manual mode
    state["mode"] = "manual"
    vs["active"]  = False
    sio.run(
    app,
    host=HOST,
    port=PORT,
    debug=False,
    allow_unsafe_werkzeug=True,
    ssl_context=("certs/cert.pem", "certs/key.pem")
    )