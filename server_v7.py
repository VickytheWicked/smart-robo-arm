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
import asyncio
import time
import serial
import serial.tools.list_ports
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
import os
import sys
import json
import re
import subprocess
import cv2
from vision_detection import detect_red_object, compute_pixel_error,detect_green_object,detect_white_object,detect_yellow_object

#For tool calling purpose

# ── Config ────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 115200
HOST         = "0.0.0.0"
PORT         = 4000

# ── Gemini Robotics ER 2 Streaming Preview ───────────────────────────────
# Uses the Live API for continuous camera input.
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-robotics-er-2-streaming-preview"
)
GEMINI_STREAM_FPS = 1.0  # Live API video input limit
GEMINI_TARGET = os.environ.get("GEMINI_TARGET", "").strip().lower()

# ── Servo calibration ────────────────────────────────────────────────────
SERVO_CONFIG = [
    {"name": "Base",       "ch": 0, "max": 540, "min": 85,  "angle": 30,  "label": "Yaw"},
    {"name": "Shoulder",   "ch": 1, "max": 520, "min": 90,  "angle": 0,   "label": "Pitch"},
    {"name": "Elbow",      "ch": 2, "max": 520, "min": 190, "angle": 180, "label": "Pitch"},
    {"name": "Wrist",      "ch": 3, "max": 400, "min": 100, "angle": 150, "label": "Pitch"},
    {"name": "WristRoll",  "ch": 4, "max": 550, "min": 100, "angle": 90,  "label": "Roll"},
    {"name": "Gripper",    "ch": 5, "max": 400, "min": 80,  "angle": 0,   "label": "Open↔Close"},
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

# Visual servoing runtime state
vs = {
    "active":        False,
    "pi_state":      "ALIGN",   # ALIGN | WAITING_DONE
    "last_cmd":      None,
    "last_cmd_time": 0.0,
    "centered_at":   0.0,
    "source":        "green",    # green | gemini
}


vs_lock = threading.Lock()

# ── Gemini streaming synchronization ─────────────────────────────────────
gemini_result_lock = threading.Lock()
latest_gemini_result = None
gemini_stream_running = False
gemini_stream_loop = None
gemini_stream_thread = None

# Rolling ESP32 log for DONE/GRAB detection by VS logic
esp_log_lines = []



PICKUP = {
    "eraser": {"area":95000,"width":100},
    "pen": {"area":95000,"width":50},
    "marker": {"area":6000,"width":40},
    "glue": {"area":5000,"width":30},
    "pencil": {"area":120000,"width":45}
}
MAX_STEP_x=5  # Maximum movement step size for servo commands
MAX_STEP_y=5  # Maximum movement step size for servo commands


def decide_command(err_x, err_y, area, class_name="unknown"):
    in_x = abs(err_x) <= DEAD_ZONE_X
    in_y = abs(err_y) <= DEAD_ZONE_Y

    pickup_cfg = PICKUP.get(str(class_name).lower())
    in_area = bool(pickup_cfg) and area > pickup_cfg["area"]

    movement_x = max(-MAX_STEP_x, min(MAX_STEP_x, int(kp_x * err_x)))
    movement_y = max(-MAX_STEP_y, min(MAX_STEP_y, int(kp_y * err_y)))

    if in_area and in_x and in_y:
        return "CENTERED:5\\nCENTERED:5\\nCENTERED:5\\nCENTERED:5\\nCENTERED:5\\nCATCH"
    elif in_x and in_y:
        return "CENTERED:" + str(MAX_STEP_y)
    elif not in_x:
        return "RIGHT:" + str(abs(movement_x)) if err_x > 0 else "LEFT:" + str(abs(movement_x))
    elif not in_y:
        return "FORWARD:" + str(abs(movement_y)) if err_y > 0 else "BACK:" + str(abs(movement_y))
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
    """30 FPS camera/display loop. Never waits for Gemini."""
    global latest_frame
    global latest_raw_frame

    cap = None

    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                add_log(
                    "Camera not available — retrying in 5s",
                    "warn",
                )
                cap = None
                time.sleep(5)
                continue

            add_log(
                "Camera opened (index 0)",
                "ok",
            )

        ret, frame = cap.read()

        if not ret:
            add_log(
                "Camera read failed — reconnecting",
                "warn",
            )
            cap.release()
            cap = None
            time.sleep(2)
            continue

        with frame_lock:
            latest_raw_frame = frame.copy()

        annotated = frame.copy()

        with gemini_result_lock:
            result = (
                latest_gemini_result.copy()
                if latest_gemini_result is not None
                else None
            )

        if result and result.get("found") and result.get("centroid"):
            cx, cy = result["centroid"]
            label = result.get("class_name") or "object"

            cv2.circle(
                annotated,
                (cx, cy),
                8,
                (0, 255, 255),
                -1,
            )

            cv2.putText(
                annotated,
                f"Gemini: {label}",
                (max(5, cx - 60), max(20, cy - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 229, 160),
                2,
                cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 55],
        )

        if ok:
            with frame_lock:
                latest_frame = buf.tobytes()

        time.sleep(0.001)


# ── Gemini Robotics ER 2 Streaming / Live API ────────────────────────────

GEMINI_SYSTEM_PROMPT = """
You are the visual perception module for a tabletop robotic arm.

Requested target:
TARGET_NAME

Inspect the latest camera image.

If the requested target is visible, output exactly one line:
TARGET_POINT <y> <x> <label>

Rules:
- y and x are integers from 0 to 1000.
- coordinates are [y, x].
- (0,0) is the top-left.
- (1000,1000) is the bottom-right.
- choose the center of the graspable portion.
- label should be short.
- output no explanation, JSON, markdown, or extra text.

If the target is not visible, output exactly:
TARGET_NOT_FOUND
"""


def parse_gemini_target(text):
    if not text:
        return None

    matches = re.findall(
        r"TARGET_POINT\s+"
        r"(\d+(?:\.\d+)?)\s+"
        r"(\d+(?:\.\d+)?)\s+"
        r"([^\r\n]+)",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    y_s, x_s, label = matches[-1]

    try:
        ny = max(0.0, min(1000.0, float(y_s)))
        nx = max(0.0, min(1000.0, float(x_s)))
    except ValueError:
        return None

    return {
        "ny": ny,
        "nx": nx,
        "label": label.strip().lower(),
    }


async def gemini_receive_loop(session):
    """Continuously receive Gemini output from the Live API."""
    global latest_gemini_result
    global latest_gemini_result_time

    try:
        while gemini_stream_running:
            async for response in session.receive():
                if not gemini_stream_running:
                    return

                server_content = getattr(
                    response,
                    "server_content",
                    None,
                )

                if not server_content:
                    continue

                model_turn = getattr(
                    server_content,
                    "model_turn",
                    None,
                )

                if not model_turn:
                    continue

                chunks = []

                for part in model_turn.parts or []:
                    part_text = getattr(part, "text", None)

                    if part_text:
                        chunks.append(part_text)

                parsed = parse_gemini_target(
                    "".join(chunks)
                )

                if not parsed:
                    continue

                with frame_lock:
                    frame = (
                        latest_raw_frame.copy()
                        if latest_raw_frame is not None
                        else None
                    )

                if frame is None:
                    continue

                h, w = frame.shape[:2]

                cx = int(
                    parsed["nx"] / 1000.0 * w
                )
                cy = int(
                    parsed["ny"] / 1000.0 * h
                )

                result = {
                    "found": True,
                    "centroid": (cx, cy),
                    "area": 0.0,
                    "annotated": frame,
                    "class_id": None,
                    "class_name": parsed["label"],
                    "confidence": 1.0,
                }

                with gemini_result_lock:
                    latest_gemini_result = result
                    latest_gemini_result_time = time.time()

                sio.emit(
                    "vision_update",
                    {
                        "found": True,
                        "centroid": [cx, cy],
                        "err_x": 0,
                        "err_y": 0,
                        "area": 0.0,
                        "class_name": parsed["label"],
                        "confidence": 1.0,
                        "source": "gemini_streaming",
                    },
                )

    except asyncio.CancelledError:
        return

    except Exception as e:
        if gemini_stream_running:
            add_log(
                f"Gemini receive loop error: {e}",
                "warn",
            )


async def gemini_heartbeat_loop(session):
    """
    Send one JPEG frame/sec and then a text instruction.

    Google documents that video frames alone do not trigger a reasoning
    turn, and Live API video input is limited to 1 FPS.
    """
    from google.genai import types

    period = 1.0 / GEMINI_STREAM_FPS

    while gemini_stream_running:
        with frame_lock:
            frame = (
                latest_raw_frame.copy()
                if latest_raw_frame is not None
                else None
            )

        if frame is not None:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 60],
            )

            if ok:
                await session.send_realtime_input(
                    video=types.Blob(
                        data=encoded.tobytes(),
                        mime_type="image/jpeg",
                    )
                )

                await session.send_realtime_input(
                    text=(
                        "[HEARTBEAT] Inspect the latest camera frame. "
                        "If the requested target is visible, output "
                        "TARGET_POINT y x label. Otherwise output "
                        "TARGET_NOT_FOUND."
                    )
                )

        await asyncio.sleep(period)


async def gemini_stream_main():
    global gemini_stream_running

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        add_log(
            "google-genai is missing. Run: "
            "pip install -U google-genai",
            "error",
        )
        return

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        add_log(
            "GEMINI_API_KEY is not set.",
            "error",
        )
        return

    client = genai.Client(
        api_key=api_key
    )

    target = (
        GEMINI_TARGET
        or "the clearest graspable object"
    )

    prompt = GEMINI_SYSTEM_PROMPT.replace(
        "TARGET_NAME",
        target,
    )

    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=types.Content(
            parts=[
                types.Part(text=prompt)
            ]
        ),
    )

    add_log(
        f"Connecting to Gemini Live: {GEMINI_MODEL}",
        "info",
    )

    try:
        async with client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config,
        ) as session:

            add_log(
                "Gemini Robotics ER 2 Live API connected",
                "ok",
            )

            receiver_task = asyncio.create_task(
                gemini_receive_loop(session)
            )

            heartbeat_task = asyncio.create_task(
                gemini_heartbeat_loop(session)
            )

            try:
                await asyncio.gather(
                    receiver_task,
                    heartbeat_task,
                )

            finally:
                for task in (
                    receiver_task,
                    heartbeat_task,
                ):
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    receiver_task,
                    heartbeat_task,
                    return_exceptions=True,
                )

    except asyncio.CancelledError:
        pass

    except Exception as e:
        add_log(
            f"Gemini Live API stopped: {e}",
            "error",
        )

    finally:
        gemini_stream_running = False


def start_gemini_stream():
    """Start exactly one dedicated asyncio loop for Gemini Live API."""
    global gemini_stream_running
    global gemini_stream_thread
    global gemini_stream_loop

    if gemini_stream_running:
        return True

    def runner():
        global gemini_stream_running
        global gemini_stream_loop

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        gemini_stream_loop = loop
        gemini_stream_running = True

        try:
            loop.run_until_complete(
                gemini_stream_main()
            )

        except Exception as e:
            add_log(
                f"Gemini streaming worker error: {e}",
                "error",
            )

        finally:
            gemini_stream_running = False
            gemini_stream_loop = None

            if not loop.is_closed():
                loop.close()

    gemini_stream_thread = threading.Thread(
        target=runner,
        daemon=True,
        name="gemini-live-api",
    )

    gemini_stream_thread.start()

    return True


def stop_gemini_stream():
    global gemini_stream_running
    gemini_stream_running = False


def visual_servo_thread():
    """
    Fast local P-controller.

    Gemini supplies a new target asynchronously. The controller never
    waits for Gemini before sending a movement command.
    """
    while True:
        with vs_lock:
            active = vs["active"]
            source = vs.get("source")

        if not active or source != "gemini":
            time.sleep(0.01)
            continue

        with gemini_result_lock:
            result = (
                latest_gemini_result.copy()
                if latest_gemini_result is not None
                else None
            )

        if not result or not result.get("found"):
            time.sleep(0.01)
            continue

        with frame_lock:
            frame = (
                latest_raw_frame.copy()
                if latest_raw_frame is not None
                else None
            )

        centroid = result.get("centroid")

        if frame is None or not centroid:
            time.sleep(0.01)
            continue

        err_x, err_y = compute_pixel_error(
            centroid,
            frame.shape,
        )

        with state_lock:
            state["detected"] = True
            state["err_x"] = err_x
            state["err_y"] = err_y

        now = time.time()

        with vs_lock:
            if not vs["active"] or vs.get("source") != "gemini":
                continue

            if now - vs["last_cmd_time"] < CMD_INTERVAL:
                time.sleep(0.005)
                continue

            cmd = decide_command(
                err_x,
                err_y,
                0.0,
                result.get(
                    "class_name",
                    "object",
                ),
            )

            # No automatic CATCH yet. Gemini streaming currently gives
            # a 2D target point, not reliable proximity/depth.
            if cmd and "\nCATCH" in cmd:
                cmd = "CENTERED:5"

            if cmd:
                vs["last_cmd"] = cmd
                vs["last_cmd_time"] = now
                vs["pi_state"] = "ALIGN"

            emit_data = {
                "pi_state": vs["pi_state"],
                "last_cmd": vs["last_cmd"],
                "source": "gemini_streaming",
            }

        if cmd:
            send_serial(cmd)

        sio.emit(
            "vs_update",
            emit_data,
        )

        time.sleep(0.005)



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
    return send_from_directory(".", "dashboard_mobile_v2.html")

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


@app.route("/api/gemini-pick", methods=["POST"])
def api_gemini_pick():
    global GEMINI_TARGET
    global latest_gemini_result

    data = request.get_json(silent=True) or {}

    enabled = bool(
        data.get("enabled", True)
    )

    target = str(
        data.get(
            "target",
            GEMINI_TARGET,
        )
    ).strip().lower()

    if enabled:
        if not os.environ.get("GEMINI_API_KEY"):
            return jsonify({
                "ok": False,
                "error": "GEMINI_API_KEY is not set.",
            }), 400

        GEMINI_TARGET = target

        with gemini_result_lock:
            latest_gemini_result = None

        with state_lock:
            state["mode"] = "visual"

        with vs_lock:
            vs["active"] = True
            vs["source"] = "gemini"
            vs["pi_state"] = "ALIGN"
            vs["last_cmd"] = None
            vs["last_cmd_time"] = 0.0
            vs["centered_at"] = 0.0

        start_gemini_stream()

        add_log(
            "Gemini Robotics ER 2 Streaming → ON"
            + (
                f" target={target}"
                if target
                else ""
            ),
            "ok",
        )

        sio.emit(
            "mode_change",
            {"mode": "visual"},
        )

        sio.emit(
            "gemini_pick",
            {
                "active": True,
                "target": target,
            },
        )

        return jsonify({
            "ok": True,
            "active": True,
            "target": target,
            "message": (
                "Gemini Robotics ER 2 Streaming enabled."
            ),
        })

    stop_gemini_stream()

    with gemini_result_lock:
        latest_gemini_result = None

    with vs_lock:
        vs["active"] = False
        vs["source"] = "green"
        vs["pi_state"] = "ALIGN"
        vs["last_cmd"] = None
        vs["last_cmd_time"] = 0.0
        vs["centered_at"] = 0.0

    with state_lock:
        state["mode"] = "manual"

    add_log(
        "Gemini Robotics ER 2 Streaming → OFF",
        "info",
    )

    sio.emit(
        "mode_change",
        {"mode": "manual"},
    )

    sio.emit(
        "gemini_pick",
        {"active": False},
    )

    return jsonify({
        "ok": True,
        "active": False,
    })


@app.route("/api/yolo-pick", methods=["POST"])
def api_yolo_pick_compat():
    return api_gemini_pick()


@app.route("/api/mode", methods=["POST"])
def api_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "manual")
    if mode != "visual":
        stop_gemini_stream()
    with state_lock:
        state["mode"] = mode
    with vs_lock:
        vs["active"]   = (mode == "visual")
        vs["source"]   = "green"
        vs["pi_state"] = "ALIGN"
        vs["last_cmd"] = None
        vs["last_cmd_time"] = 0.0
        vs["centered_at"] = 0.0
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
    stop_gemini_stream()
    """Reset the visual servoing state."""
    with vs_lock:
        vs["active"]   = False
        vs["source"]   = "green"
        vs["pi_state"] = "ALIGN"
        vs["last_cmd"] = None
        vs["last_cmd_time"] = 0.0
        vs["centered_at"] = 0.0
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
    print(f"[DASHBOARD] Starting on http://0.0.0.0:{PORT}")
    threading.Thread(target=serial_reader_thread, daemon=True).start()

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=visual_servo_thread, daemon=True).start()
   
   
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