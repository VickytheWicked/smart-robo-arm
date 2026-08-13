"""
app.py  —  Raspberry Pi 5 Dashboard Server
Flask + Flask-SocketIO bridge between browser dashboard and ESP32.

Run:  python app.py
Open: http://raspberrypi.local:5000   (or http://<pi-ip>:5000)

Install:
  pip install flask flask-socketio pyserial opencv-python numpy
"""

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
from vision_detection import detect_red_object, compute_pixel_error

# ── Config ────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 115200
HOST         = "0.0.0.0"
PORT         = 3000
AGENT_LOG    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_log.jsonl")

# Visual servoing
DEAD_ZONE_X   = 30    # px
DEAD_ZONE_Y   = 30    # px
CMD_INTERVAL  = 0.08  # seconds between commands
CENTERED_HOLD = 0.5   # seconds to wait after CENTERED before re-evaluating
VS_TIMEOUT    = 15.0  # seconds before auto-reset if grab never completes

# Servo calibration — mirrors all_servo[] in scripted/main.cpp
SERVO_CONFIG = [
    {"name": "Base",      "ch": 0, "max": 540, "min": 85,  "angle": 89,  "label": "Yaw"},
    {"name": "Shoulder",  "ch": 1, "max": 520, "min": 90,  "angle": 115, "label": "Pitch"},
    {"name": "Elbow",     "ch": 2, "max": 520, "min": 190, "angle": 90,  "label": "Pitch"},
    {"name": "Wrist",     "ch": 3, "max": 400, "min": 100, "angle": 80,  "label": "Pitch"},
    {"name": "WristRoll", "ch": 4, "max": 550, "min": 100, "angle": 90,  "label": "Roll"},
    {"name": "Gripper",   "ch": 5, "max": 400, "min": 80,  "angle": 0,   "label": "Open↔Close"},
]

# Scripted keyframes from scripted/main.cpp (servo_index, angle pairs)
SEQUENCES = {
    "original": [{"s": 0, "a": 89}, {"s": 2, "a": 90}, {"s": 1, "a": 115}, {"s": 3, "a": 80}, {"s": 4, "a": 90}],
    "moveX":    [{"s": 0, "a": 95}, {"s": 2, "a": 45}, {"s": 1, "a": 90},  {"s": 3, "a": 90}, {"s": 4, "a": 90}],
    "lift1":    [{"s": 0, "a": 90}, {"s": 2, "a": 71}, {"s": 1, "a": 116}, {"s": 3, "a": 80}, {"s": 4, "a": 90}],
    "moveY":    [{"s": 0, "a": 14}, {"s": 2, "a": 71}, {"s": 1, "a": 116}, {"s": 3, "a": 80}, {"s": 4, "a": 90}],
    "place":    [{"s": 0, "a": 14}, {"s": 2, "a": 133},{"s": 1, "a": 50},  {"s": 3, "a": 80}, {"s": 4, "a": 90}],
    "lift2":    [{"s": 0, "a": 14}, {"s": 2, "a": 133},{"s": 1, "a": 116}, {"s": 3, "a": 83}, {"s": 4, "a": 90}],
}

# ── App ───────────────────────────────────────────────────────────────────
app    = Flask(__name__, static_folder=".", template_folder=".")
sio    = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state
state = {
    "angles":    [c["angle"] for c in SERVO_CONFIG],
    "distance":  0.0,
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

# ── Agent process state ────────────────────────────────────────────────────────
agent_proc = {"process": None, "pid": None, "running": False}


def decide_command(err_x, err_y):
    in_x = abs(err_x) <= DEAD_ZONE_X
    in_y = abs(err_y) <= DEAD_ZONE_Y
    if in_x and in_y:
        return "CENTERED"
    if not in_x:
        return "RIGHT" if err_x > 0 else "LEFT"
    if not in_y:
        return "FORWARD" if err_y > 0 else "BACK"
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
                if line.startswith("Distance:"):
                    try:
                        dist = float(line.split(":")[1].strip())
                        with state_lock:
                            state["distance"] = round(dist, 1)
                        sio.emit("sensor", {"distance": dist})
                    except ValueError:
                        pass

                # Parse state
                if "state" in line.lower() or "ALIGN" in line or "GRAB" in line or "DONE" in line:
                    for s in ["ALIGN", "APPROACH", "GRAB", "DONE"]:
                        if s in line:
                            with state_lock:
                                state["arm_state"] = s
                            sio.emit("arm_state", {"state": s})
                            break

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

        result    = detect_red_object(frame)
        annotated = result["annotated"]

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame = buf.tobytes()

        err_x, err_y = 0, 0
        if result["found"] and result["centroid"]:
            err_x, err_y = compute_pixel_error(result["centroid"], frame.shape)

        with state_lock:
            state["detected"] = result["found"]
            state["err_x"]    = err_x
            state["err_y"]    = err_y

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

                if result["found"] and (now - vs["last_cmd_time"]) >= CMD_INTERVAL:
                    if vs["pi_state"] == "ALIGN":
                        cmd = decide_command(err_x, err_y)
                        if cmd == "CENTERED":
                            vs["pi_state"]      = "WAITING_DONE"
                            vs["centered_at"]   = now
                            vs["last_cmd"]      = "CENTERED"
                            vs["last_cmd_time"] = now
                            cmd_to_send = "CENTERED"
                            vs_log = ("[VS] Centered — waiting for grab", "ok")
                        elif cmd and cmd != vs["last_cmd"]:
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
    return send_from_directory(".", "dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with state_lock:
        s = dict(state)
    s["agent_running"] = agent_proc["running"]
    return jsonify(s)


@app.route("/api/command", methods=["POST"])
def api_command():
    """Send a raw command string to ESP32 (LEFT/RIGHT/FORWARD/BACK/CENTERED)."""
    data = request.get_json()
    cmd  = data.get("cmd", "").upper().strip()
    VALID = {"LEFT", "RIGHT", "FORWARD", "BACK", "CENTERED"}
    if cmd not in VALID:
        return jsonify({"ok": False, "error": f"Unknown command: {cmd}"}), 400
    ok = send_serial(cmd)
    return jsonify({"ok": ok})


@app.route("/api/gripper/<action>", methods=["POST"])
def api_gripper(action):
    """Open or close gripper by sending matching serial command."""
    # Visual servo firmware responds to FORWARD/BACK/LEFT/RIGHT for alignment
    # Gripper is triggered internally by the state machine; we expose a manual override
    # by encoding as a comment — in manual mode the Pi can send direct angle commands
    if action == "open":
        add_log("Gripper OPEN (manual)", "warn")
        return jsonify({"ok": True, "note": "Gripper open — ESP32 state machine controls gripper in AUTO mode"})
    elif action == "close":
        add_log("Gripper CLOSE (manual)", "warn")
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400


@app.route("/api/sequence/<name>", methods=["POST"])
def api_sequence(name):
    """Trigger a named keyframe sequence (info only — scripted mode runs on ESP32 via button)."""
    if name not in SEQUENCES and name not in ("action", "home"):
        return jsonify({"ok": False, "error": "Unknown sequence"}), 404
    add_log(f"Sequence '{name}' triggered — ensure scripted firmware is flashed", "warn")
    # In scripted mode the ESP32 runs action() on button press.
    # With visual_servo firmware, you can send CENTERED to start the approach.
    if name == "home":
        send_serial("BACK")   # safe approximate home move
    return jsonify({"ok": True, "sequence": name})


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


@app.route("/api/ports")
def api_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports})


# ── Agent control endpoints ────────────────────────────────────────────────────

@app.route("/api/agent/log")
def api_agent_log():
    """Return last 20 lines of agent_log.jsonl as a JSON array."""
    if not os.path.exists(AGENT_LOG):
        return jsonify([])
    try:
        with open(AGENT_LOG, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-20:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return jsonify(entries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/start", methods=["POST"])
def api_agent_start():
    """Launch agent.py as a subprocess. Idempotent if already running."""
    if agent_proc["running"] and agent_proc["process"] and agent_proc["process"].poll() is None:
        return jsonify({"ok": True, "pid": agent_proc["pid"], "note": "already running"})

    data    = request.get_json(silent=True) or {}
    model   = data.get("model", "moondream")
    no_voice = data.get("no_voice", False)
    verbose  = data.get("verbose", False)

    script  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myenv", "bin", "python")
    py      = venv_py if os.path.exists(venv_py) else sys.executable
    cmd     = [py, script, "--headless", "--no-voice", "--model", model]
    if verbose:
        cmd.append("--verbose")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        agent_proc["process"] = proc
        agent_proc["pid"]     = proc.pid
        agent_proc["running"] = True
        add_log(f"[AGENT] Started pid={proc.pid} model={model}", "ok")
        return jsonify({"ok": True, "pid": proc.pid})
    except Exception as e:
        add_log(f"[AGENT] Start failed: {e}", "error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/agent/stop", methods=["POST"])
def api_agent_stop():
    """Kill the running agent subprocess."""
    proc = agent_proc.get("process")
    if proc is None or proc.poll() is not None:
        agent_proc["running"] = False
        agent_proc["pid"]     = None
        return jsonify({"ok": True, "note": "not running"})
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        agent_proc["running"] = False
        agent_proc["pid"]     = None
        add_log("[AGENT] Stopped", "warn")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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


@sio.on("vision_data")
def on_vision_data(data):
    """visual_servoing.py pushes detection data here."""
    with state_lock:
        state["detected"] = data.get("found", False)
        state["err_x"]    = data.get("err_x", 0)
        state["err_y"]    = data.get("err_y", 0)
    sio.emit("vision_update", data)


# ── Main ───────────────────────────────────────────────────────────────────

def agent_log_tailer():
    """Poll agent_log.jsonl every second for new lines and emit agent_update events."""
    last_pos = 0
    while True:
        try:
            if os.path.exists(AGENT_LOG):
                with open(AGENT_LOG, "r") as f:
                    f.seek(last_pos)
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                        except Exception:
                            continue
                        dec = entry.get("decision", {})
                        sio.emit("agent_update", {
                            "speech":     dec.get("speech",         ""),
                            "command":    dec.get("command",        "NONE"),
                            "confidence": dec.get("confidence",     0.0),
                            "note":       dec.get("note",           ""),
                            "target":     dec.get("target_visible", False),
                            "timestamp":  entry.get("timestamp",    ""),
                            "model":      entry.get("model",        ""),
                        })
                    last_pos = f.tell()
        except Exception:
            pass

        # Check if agent process is still alive and update running flag
        proc = agent_proc.get("process")
        if proc is not None and agent_proc["running"] and proc.poll() is not None:
            agent_proc["running"] = False
            agent_proc["pid"]     = None
            add_log("[AGENT] Process exited", "warn")

        time.sleep(1)


if __name__ == "__main__":
    print(f"[DASHBOARD] Starting on http://0.0.0.0:{PORT}")
    threading.Thread(target=serial_reader_thread, daemon=True).start()
    threading.Thread(target=camera_thread,        daemon=True).start()
    threading.Thread(target=agent_log_tailer,     daemon=True).start()
    # Always boot in manual mode
    state["mode"] = "manual"
    vs["active"]  = False
    sio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
