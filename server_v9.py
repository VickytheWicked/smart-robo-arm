"""
server_v9.py — Universal Target Visual Servoing & Dashboard Server
Flask + Flask-SocketIO bridge between web/voice clients and ESP32 robotic arm.

Features:
- Any-object pick via Gemini Robotics ER-2 (gemini-robotics-er-2-preview) + OpenCV CSRT 30 FPS tracking
- Anti-drift heuristic validation from crst.py (jump, area, aspect-ratio limits)
- Closed-loop proportional visual servoing to center and grasp
- Full REST API compatible with robot/arm_v2.py and gemini_server_v2.py
- Real-time SocketIO telemetry, live MJPEG video feed, and serial management
"""

import cv2
import json
import logging
import os
import re
import serial
import serial.tools.list_ports
import threading
import time

from flask import Flask, jsonify, render_template, request, Response, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from google import genai
from google.genai import types

from vision_detection import (
    detect_color_object,
    extract_color_from_string,
    SUPPORTED_COLORS,
)

# Load environment variables from .env
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT       = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD_RATE         = int(os.environ.get("BAUD_RATE", "115200"))
HOST              = os.environ.get("HOST", "0.0.0.0")
PORT              = int(os.environ.get("PORT", "3000"))
DRY_RUN           = os.environ.get("DRY_RUN", "False").lower() in ("1", "true")

GEMINI_MODEL      = os.environ.get("GEMINI_ROBOTICS_MODEL", "gemini-robotics-er-2-preview")
DEFAULT_TARGET    = os.environ.get("DEFAULT_TARGET", None)
GEMINI_INTERVAL         = float(os.environ.get("GEMINI_INTERVAL", "1.3"))
COMMAND_INTERVAL        = float(os.environ.get("COMMAND_INTERVAL", "0.6"))
COLOR_COMMAND_INTERVAL  = float(os.environ.get("COLOR_COMMAND_INTERVAL", "0.2"))  # Dedicated interval for OpenCV color detection

CAMERA_INDEX      = int(os.environ.get("CAMERA_INDEX", "1"))
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
CAMERA_FPS        = 30
CENTER_X          = FRAME_WIDTH // 2
CENTER_Y          = FRAME_HEIGHT // 2

DEAD_ZONE_X       = 60  # pixels
DEAD_ZONE_Y       = 60  # pixels
KP_X              = 0.035
KP_Y              = 0.035
MAX_STEP_X        = 5
MAX_STEP_Y        = 5

# Anti-drift validation parameters (from crst.py)
MAX_TRACK_JUMP            = 45.0   # Max centroid movement px between frames
MAX_AREA_RATIO            = 2.5    # Max area expansion/contraction ratio
MAX_ASPECT_RATIO_CHANGE   = 1.35   # Max aspect ratio distortion
TRACK_BAD_FRAME_LIMIT     = 3      # Consecutive bad frames before tracker reset
TRACKER_MAX_AGE           = 0.6    # Stale tracker timeout in seconds

# Servo calibration — mirrors all_servo[] in ESP32 firmware
SERVO_CONFIG = [
    {"name": "Base",      "ch": 0, "max": 540, "min": 85,  "angle": 30,  "label": "Yaw"},
    {"name": "Shoulder",  "ch": 1, "max": 520, "min": 90,  "angle": 0,   "label": "Pitch"},
    {"name": "Elbow",     "ch": 2, "max": 520, "min": 190, "angle": 180, "label": "Pitch"},
    {"name": "Wrist",     "ch": 3, "max": 400, "min": 100, "angle": 150, "label": "Pitch"},
    {"name": "WristRoll", "ch": 4, "max": 550, "min": 100, "angle": 90,  "label": "Roll"},
    {"name": "Gripper",   "ch": 5, "max": 400, "min": 80,  "angle": 0,   "label": "Open↔Close"},
]

# ── App & SocketIO ────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", template_folder=".")
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global State ──────────────────────────────────────────────────────────────
state = {
    "angles": [c["angle"] for c in SERVO_CONFIG],
    "mode": "manual",        # manual | visual
    "arm_state": "IDLE",
    "target": DEFAULT_TARGET,
    "detected": False,
    "err_x": 0,
    "err_y": 0,
    "log": [],
    "esp_connected": False,
    "vision_mode": "idle",   # idle | gemini | color
    "active_color": None,    # red | green | yellow | blue | white
}
state_lock = threading.RLock()

# Vision Mode & Color Tracker State
vision_mode = "idle"            # "idle" | "gemini" | "color"
active_color = None             # "red" | "green" | "yellow" | "blue" | "white"
active_target = DEFAULT_TARGET
vision_lock = threading.RLock()

# Serial connection state
ser_conn = None
serial_lock = threading.RLock()
esp_log_lines = []

# Camera & Streaming state
latest_camera_frame = None
latest_encoded_jpeg = None
frame_lock = threading.RLock()

# Tracker State
tracker = None
tracker_bbox = None             # (x, y, w, h)
tracker_found = False
tracker_last_update = 0.0
tracker_previous_bbox = None
tracker_bad_frame_count = 0
tracker_last_gemini_version = 0
tracker_lock = threading.RLock()

# Visual Servo Control State
vs = {
    "active": False,
    "target": DEFAULT_TARGET,
    "pi_state": "ALIGN",        # ALIGN | CENTERED | TO_CATCH
    "last_cmd": None,
    "last_cmd_time": 0.0,
    "centered_count": 0,
}
vs_lock = threading.RLock()


# ── Tracker Factory ───────────────────────────────────────────────────────────
def create_tracker():
    """Instantiate OpenCV CSRT tracker with fallback to legacy API."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV CSRT tracker is unavailable. Please install opencv-contrib-python.")


# ── Anti-Drift Tracker Validation (crst.py logic) ──────────────────────────────
def validate_tracker_update(new_bbox, previous=None):
    if previous is None:
        with tracker_lock:
            previous = tracker_previous_bbox

    if previous is None:
        return True, "FIRST"

    px, py, pw, ph = previous
    nx, ny, nw, nh = new_bbox

    # 1. Centroid Jump Check
    previous_cx = px + pw / 2.0
    previous_cy = py + ph / 2.0
    new_cx = nx + nw / 2.0
    new_cy = ny + nh / 2.0

    jump = ((new_cx - previous_cx) ** 2 + (new_cy - previous_cy) ** 2) ** 0.5
    if jump > MAX_TRACK_JUMP:
        return False, f"CENTROID_JUMP={jump:.1f}px"

    # 2. Area Ratio Jump Check
    previous_area = max(1.0, pw * ph)
    new_area = max(1.0, nw * nh)
    area_ratio = new_area / previous_area
    if area_ratio > MAX_AREA_RATIO or area_ratio < (1.0 / MAX_AREA_RATIO):
        return False, f"AREA_JUMP={area_ratio:.2f}x"

    # 3. Aspect Ratio Distortion Check
    previous_aspect = pw / max(ph, 1.0)
    new_aspect = nw / max(nh, 1.0)
    aspect_ratio = new_aspect / max(previous_aspect, 0.001)
    if aspect_ratio > MAX_ASPECT_RATIO_CHANGE or aspect_ratio < (1.0 / MAX_ASPECT_RATIO_CHANGE):
        return False, f"ASPECT_JUMP={aspect_ratio:.2f}x"

    return True, "VALID"


# ── Gemini Robotics Target Localizer ──────────────────────────────────────────
class GeminiRoboticsTracker:
    """
    Asynchronous VLM worker querying gemini-robotics-er-2-preview
    to locate whatever target object the user requests.
    """

    def __init__(self, target=DEFAULT_TARGET):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[GEMINI] Notice: GEMINI_API_KEY environment variable not currently set.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
        self.target = target
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_result = None
        self.result_lock = threading.Lock()
        self.result_version = 0

        self.running = False
        self.worker = None
        self.last_request_time = 0.0

    def set_target(self, target):
        with self.result_lock:
            if target is None or str(target).strip().lower() in ("none", "", "null", "false"):
                self.target = None
                self.latest_result = None
                self.result_version += 1
                add_log("[GEMINI] Target cleared — frame processing stopped", "info")
            else:
                self.target = str(target).strip()
                self.latest_result = None
                self.result_version += 1
                add_log(f"[GEMINI] Target object updated → '{self.target}'", "ok")
        with tracker_lock:
            global tracker, tracker_found, tracker_bbox, tracker_previous_bbox, tracker_bad_frame_count
            tracker = None
            tracker_found = False
            tracker_bbox = None
            tracker_previous_bbox = None
            tracker_bad_frame_count = 0

    def update_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def get_result(self):
        with self.result_lock:
            if self.latest_result is None:
                return None
            return self.latest_result.copy()

    def get_version(self):
        with self.result_lock:
            return self.result_version

    def build_prompt(self):
        return f"""
You are the vision system for a robotic arm.

The robot needs to pick:
"{self.target}"

Locate ONLY this target object.
Ignore every other object.

Return ONLY valid JSON:
{{
    "label": "{self.target}",
    "box_2d": [ymin, xmin, ymax, xmax]
}}

Coordinates are normalized from 0 to 1000.
box_2d format: [ymin, xmin, ymax, xmax]

IMPORTANT:
- The box must tightly surround ONLY the target.
- Do not include the table or background.
- Do not include nearby objects.
- Do not merge objects.
- Do not invent the object.

If the target is not visible, return:
{{
    "label": "{self.target}",
    "box_2d": null
}}

Do not explain anything.
Do not use markdown.
Return JSON only.
"""

    def parse_json(self, text):
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def process_result(self, raw, frame):
        height, width = frame.shape[:2]
        box = raw.get("box_2d")
        if not box or not isinstance(box, list) or len(box) != 4:
            return {
                "found": False,
                "centroid": None,
                "area": 0.0,
                "bbox": None,
                "class_name": self.target,
                "confidence": 0.0,
            }

        try:
            ymin, xmin, ymax, xmax = [float(v) for v in box]
        except (ValueError, TypeError):
            return None

        x1 = max(0, min(width - 1, int(xmin / 1000.0 * width)))
        y1 = max(0, min(height - 1, int(ymin / 1000.0 * height)))
        x2 = max(0, min(width - 1, int(xmax / 1000.0 * width)))
        y2 = max(0, min(height - 1, int(ymax / 1000.0 * height)))

        if x2 <= x1 or y2 <= y1:
            return None

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        area = float(w * h)

        return {
            "found": True,
            "centroid": (cx, cy),
            "area": area,
            "bbox": (int(x1), int(y1), int(w), int(h)),
            "class_name": self.target,
            "confidence": 1.0,
        }

    def detect(self, frame):
        if not self.client:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise RuntimeError("Could not encode frame for Gemini.")

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
                self.build_prompt(),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
        return self.parse_json(response.text)

    def worker_thread(self):
        print(f"[GEMINI] Target localizer started for model '{GEMINI_MODEL}'")
        while self.running:
            if not self.target:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - self.last_request_time < GEMINI_INTERVAL:
                time.sleep(0.01)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            self.last_request_time = now
            try:
                raw = self.detect(frame)
                if raw is None:
                    continue
                result = self.process_result(raw, frame)
                if result is None:
                    continue

                with self.result_lock:
                    self.latest_result = result
                    self.result_version += 1
                    ver = self.result_version

                if result["found"]:
                    add_log(f"[GEMINI] #{ver} Found '{self.target}' @ {result['centroid']} area={int(result['area'])}", "ok")
                else:
                    add_log(f"[GEMINI] #{ver} '{self.target}' not found in view", "info")

            except Exception as e:
                add_log(f"[GEMINI] Detection error: {e}", "warn")
                time.sleep(1.0)

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker = threading.Thread(target=self.worker_thread, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False


gemini_tracker = GeminiRoboticsTracker(target=DEFAULT_TARGET)


# ── Serial Communication ──────────────────────────────────────────────────────
def get_serial():
    global ser_conn
    if DRY_RUN:
        print("[SERIAL] DRY_RUN mode enabled — commands logged without sending.")
        return None
    try:
        ser_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(1.5)
        add_log(f"Serial connected: {SERIAL_PORT} @ {BAUD_RATE}", "ok")
        with state_lock:
            state["esp_connected"] = True
        sio.emit("status", {"esp": True})
        return ser_conn
    except serial.SerialException as e:
        add_log(f"Serial connection failed: {e}", "error")
        with state_lock:
            state["esp_connected"] = False
        sio.emit("status", {"esp": False})
        return None


def serial_reader_thread():
    global ser_conn
    if DRY_RUN:
        return
    while True:
        if ser_conn is None or not ser_conn.is_open:
            ser_conn = get_serial()
            if ser_conn is None:
                time.sleep(3.0)
                continue
        try:
            if ser_conn.in_waiting:
                line = ser_conn.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    esp_log_lines.append(line)
                    if len(esp_log_lines) > 200:
                        esp_log_lines.pop(0)
                    if "DONE" in line or "GRAB" in line:
                        with vs_lock:
                            if vs["pi_state"] == "TO_CATCH":
                                vs["pi_state"] = "ALIGN"
                                add_log(f"[ESP32] Grab completed: {line}", "ok")
                    print(f"[ESP32] {line}")
        except (serial.SerialException, OSError) as e:
            add_log(f"Serial read error: {e}", "warn")
            ser_conn = None
        time.sleep(0.005)


def send_serial(cmd: str) -> bool:
    global ser_conn
    if DRY_RUN:
        print(f"[SERIAL DRY-RUN] → {cmd}")
        return True
    if ser_conn is None or not ser_conn.is_open:
        add_log(f"Serial not connected — command skipped: {cmd}", "warn")
        return False
    try:
        with serial_lock:
            ser_conn.write((cmd + "\n").encode())
            ser_conn.flush()
        print(f"[SERIAL SENT] → {cmd}")
        return True
    except (serial.SerialException, OSError) as e:
        add_log(f"Serial send error: {e}", "error")
        ser_conn = None
        return False


def add_log(msg: str, level="info"):
    entry = {"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    with state_lock:
        state["log"].append(entry)
        if len(state["log"]) > 500:
            state["log"].pop(0)
    sio.emit("log", entry)


# ── Visual Servo Control Decision (from crst.py) ──────────────────────────────
def decide_command(err_x, err_y, area=0.0):
    """
    Computes proportional velocity adjustment toward image center.
    When centered within dead zone, returns CENTERED:3 which signals
    the ESP32 to descend to calibrated table coordinate and grasp.
    """
    in_x = abs(err_x) <= DEAD_ZONE_X
    in_y = abs(err_y) <= DEAD_ZONE_Y

    movement_x = int(KP_X * err_x)
    movement_y = int(KP_Y * err_y)

    movement_x = max(-MAX_STEP_X, min(MAX_STEP_X, movement_x))
    movement_y = max(-MAX_STEP_Y, min(MAX_STEP_Y, movement_y))

    if movement_x == 0:
        movement_x = 1
    if movement_y == 0:
        movement_y = 1

    # 1. Align Horizontal Axis
    if not in_x:
        if err_x > 0:
            return f"RIGHT:{max(1, movement_x - 1)}"
        return f"LEFT:{max(1, abs(movement_x) - 1)}"

    # 2. Align Vertical Pitch Axis
    if not in_y:
        if err_y > 0:
            return f"FORWARD:{max(1, movement_y)}"
        return f"BACK:{max(1, abs(movement_y))}"

    # 3. Target is Centered → trigger calibrated descent & grab
    return f"CENTERED:{max(1, 2)}"


# ── Camera & CSRT Tracking Thread ─────────────────────────────────────────────
def camera_thread():
    global latest_camera_frame, latest_encoded_jpeg
    global tracker, tracker_bbox, tracker_found, tracker_last_update
    global tracker_previous_bbox, tracker_bad_frame_count, tracker_last_gemini_version

    active_camera_index = CAMERA_INDEX
    cap = None
    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()

            candidate_indices = [active_camera_index] + [i for i in (1, 2, 0, 3) if i != active_camera_index]
            for idx in candidate_indices:
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        active_camera_index = idx
                        break
                test_cap.release()

            if cap is None or not cap.isOpened():
                add_log(f"Camera not accessible — retrying in 4s", "warn")
                time.sleep(4.0)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            add_log(f"Camera opened on index {active_camera_index} (640x480 @ 30 FPS)", "ok")

        ret, frame = cap.read()
        if not ret:
            add_log("Camera frame read failed — reconnecting", "warn")
            cap.release()
            cap = None
            time.sleep(2.0)
            continue

        annotated = frame.copy()

        with vision_lock:
            cur_mode = vision_mode
            cur_color = active_color
            cur_target = active_target

        if cur_mode == "color" and cur_color:
            # High-speed local HSV color detection via vision_detection.py (e.g. lego brick)
            color_res = detect_color_object(frame, cur_color)
            current_found = color_res.get("found", False)
            current_bbox = color_res.get("bbox")
            color_mask = color_res.get("mask")

            with tracker_lock:
                tracker_found = current_found
                tracker_bbox = current_bbox
                if current_found:
                    tracker_last_update = time.time()

            cx, cy, err_x, err_y, area = None, None, 0, 0, 0.0
            target_name = cur_target or f"{cur_color} object"

            # Crosshairs at center target
            cv2.drawMarker(annotated, (CENTER_X, CENTER_Y), (80, 80, 80), cv2.MARKER_CROSS, 24, 1)
            cv2.rectangle(
                annotated,
                (CENTER_X - DEAD_ZONE_X, CENTER_Y - DEAD_ZONE_Y),
                (CENTER_X + DEAD_ZONE_X, CENTER_Y + DEAD_ZONE_Y),
                (60, 60, 60),
                1,
                cv2.LINE_AA
            )

            if current_found and current_bbox is not None:
                bx, by, bw, bh = current_bbox
                cx = bx + bw // 2
                cy = by + bh // 2
                err_x = cx - CENTER_X
                err_y = cy - CENTER_Y
                area = float(bw * bh)

                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 229, 160), 2)
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
                cv2.line(annotated, (CENTER_X, CENTER_Y), (cx, cy), (0, 180, 255), 1, cv2.LINE_AA)

                label = f"{target_name} [{cur_color.upper()} HSV 30fps]"
                cv2.putText(annotated, label, (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 160), 2)
                cv2.putText(annotated, f"err: ({err_x:+d}, {err_y:+d}) area={int(area)}", (bx, by + bh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                status_text = f"Searching for: '{target_name}' [{cur_color.upper()} HSV]..."
                cv2.putText(annotated, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 255), 2)

            # Inset binary mask preview on top-right corner
            if color_mask is not None:
                try:
                    ms = cv2.resize(color_mask, (120, 90))
                    annotated[8:98, FRAME_WIDTH - 128:FRAME_WIDTH - 8] = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(annotated, (FRAME_WIDTH - 128, 8), (FRAME_WIDTH - 8, 98), (80, 80, 80), 1)
                except Exception:
                    pass

            # Update telemetry & UI
            with state_lock:
                state["detected"] = bool(current_found)
                state["err_x"] = err_x
                state["err_y"] = err_y
                state["target"] = target_name
                state["vision_mode"] = "color"
                state["active_color"] = cur_color

            sio.emit("vision_update", {
                "found": bool(current_found),
                "centroid": [cx, cy] if cx is not None else None,
                "err_x": err_x,
                "err_y": err_y,
                "area": area,
                "target": target_name,
                "source": "vision_detection_hsv",
                "color": cur_color
            })

        else:
            # Universal object localization via Gemini ER-2 + CSRT
            gemini_tracker.update_frame(frame)

            # Check for fresh Gemini detection
            gemini_result = gemini_tracker.get_result()
            gemini_version = gemini_tracker.get_version()

            with tracker_lock:
                # Re-seed CSRT tracker if Gemini returned a new detection
                if gemini_result and gemini_result["found"] and gemini_version != tracker_last_gemini_version:
                    gx, gy, gw, gh = gemini_result["bbox"]
                    try:
                        new_tracker = create_tracker()
                        new_tracker.init(frame, (gx, gy, gw, gh))
                        tracker = new_tracker
                        tracker_bbox = (gx, gy, gw, gh)
                        tracker_found = True
                        tracker_last_update = time.time()
                        tracker_previous_bbox = (gx, gy, gw, gh)
                        tracker_bad_frame_count = 0
                        tracker_last_gemini_version = gemini_version
                        add_log(f"[TRACKER] CSRT initialized for '{gemini_tracker.target}' @ ({gx},{gy},{gw},{gh})", "ok")
                    except Exception as e:
                        add_log(f"[TRACKER] Failed to create CSRT tracker: {e}", "warn")

                # Update CSRT tracker frame by frame
                if tracker is not None and tracker_found:
                    success, new_bbox = tracker.update(frame)
                    if success:
                        bx, by, bw, bh = [int(v) for v in new_bbox]
                        valid, reason = validate_tracker_update((bx, by, bw, bh), previous=tracker_previous_bbox)
                        if valid:
                            tracker_bbox = (bx, by, bw, bh)
                            tracker_previous_bbox = (bx, by, bw, bh)
                            tracker_last_update = time.time()
                            tracker_bad_frame_count = 0
                        else:
                            tracker_bad_frame_count += 1
                            if tracker_bad_frame_count >= TRACK_BAD_FRAME_LIMIT:
                                tracker_found = False
                                tracker = None
                                tracker_bbox = None
                                tracker_previous_bbox = None
                                tracker_bad_frame_count = 0
                                add_log(f"[TRACKER] Lost tracking ({reason}) — re-triggering Gemini", "warn")
                    else:
                        tracker_found = False
                        tracker = None
                        tracker_bbox = None
                        tracker_bad_frame_count = 0

                current_found = tracker_found
                current_bbox = tracker_bbox

            # Visualization Overlays
            cx, cy, err_x, err_y, area = None, None, 0, 0, 0.0
            target_name = gemini_tracker.target

            # Crosshairs at center target
            cv2.drawMarker(annotated, (CENTER_X, CENTER_Y), (80, 80, 80), cv2.MARKER_CROSS, 24, 1)
            cv2.rectangle(
                annotated,
                (CENTER_X - DEAD_ZONE_X, CENTER_Y - DEAD_ZONE_Y),
                (CENTER_X + DEAD_ZONE_X, CENTER_Y + DEAD_ZONE_Y),
                (60, 60, 60),
                1,
                cv2.LINE_AA
            )

            if current_found and current_bbox is not None:
                bx, by, bw, bh = current_bbox
                cx = bx + bw // 2
                cy = by + bh // 2
                err_x = cx - CENTER_X
                err_y = cy - CENTER_Y
                area = float(bw * bh)

                # Draw target bounding box
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 229, 160), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
                cv2.line(annotated, (CENTER_X, CENTER_Y), (cx, cy), (0, 180, 255), 1, cv2.LINE_AA)

                label = f"{target_name} [CSRT 30fps]"
                cv2.putText(annotated, label, (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 160), 2)
                cv2.putText(annotated, f"err: ({err_x:+d}, {err_y:+d})", (bx, by + bh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            elif not target_name:
                cv2.putText(annotated, "Awaiting target (speak to Lucy or POST /api/pick)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            else:
                status_text = f"Searching for: '{target_name}'..."
                cv2.putText(annotated, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 255), 2)

            # Update telemetry & UI
            with state_lock:
                state["detected"] = bool(current_found)
                state["err_x"] = err_x
                state["err_y"] = err_y
                state["target"] = target_name
                state["vision_mode"] = "gemini" if target_name else "idle"
                state["active_color"] = None

            sio.emit("vision_update", {
                "found": bool(current_found),
                "centroid": [cx, cy] if cx is not None else None,
                "err_x": err_x,
                "err_y": err_y,
                "area": area,
                "target": target_name,
                "source": "csrt_gemini",
            })

        # JPEG encoding for video stream
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        with frame_lock:
            latest_encoded_jpeg = buf.tobytes()

        time.sleep(0.01)


# ── Visual Servoing Loop ──────────────────────────────────────────────────────
def visual_servo_thread():
    global last_command_time
    print("[VS] Visual servoing loop running")

    while True:
        with vs_lock:
            active = vs["active"]

        if not active:
            time.sleep(0.05)
            continue

        with tracker_lock:
            found = tracker_found
            bbox = tracker_bbox
            last_up = tracker_last_update

        if not found or bbox is None:
            time.sleep(0.02)
            continue

        age = time.time() - last_up
        if age > TRACKER_MAX_AGE:
            time.sleep(0.02)
            continue

        bx, by, bw, bh = bbox
        cx = bx + bw // 2
        cy = by + bh // 2
        err_x = cx - CENTER_X
        err_y = cy - CENTER_Y
        area = float(bw * bh)

        with vision_lock:
            active_interval = COLOR_COMMAND_INTERVAL if vision_mode == "color" else COMMAND_INTERVAL

        now = time.time()
        with vs_lock:
            if now - vs["last_cmd_time"] < active_interval:
                time.sleep(0.01)
                continue

            cmd = decide_command(err_x, err_y, area)
            if cmd:
                vs["last_cmd"] = cmd
                vs["last_cmd_time"] = now

                if "CENTERED" in cmd:
                    vs["pi_state"] = "CENTERED"
                    vs["centered_count"] += 1
                else:
                    vs["pi_state"] = "ALIGN"
                    vs["centered_count"] = 0

                vs_emit = {
                    "pi_state": vs["pi_state"],
                    "last_cmd": vs["last_cmd"],
                    "target": vs["target"],
                    "active": True
                }
            else:
                vs_emit = None

        if cmd:
            send_serial(cmd)
            add_log(f"[VS] Command → {cmd} (err_x={err_x}, err_y={err_y})", "info")

        if vs_emit:
            sio.emit("vs_update", vs_emit)

        time.sleep(0.01)


def gen_frames():
    """Video stream frame generator for /video_feed."""
    while True:
        with frame_lock:
            frame_bytes = latest_encoded_jpeg
        if frame_bytes is None:
            time.sleep(0.04)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.033)


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.route("/")
def index():
    if os.path.exists("dashboard_mobile_v3.html"):
        return send_from_directory(".", "dashboard_mobile_v3.html")
    if os.path.exists("dashboard_mobile_v2.html"):
        return send_from_directory(".", "dashboard_mobile_v2.html")
    return jsonify({"service": "server_v9", "status": "running"})


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/training/status")
def api_training_status():
    return jsonify({"status": "idle", "message": "Universal target picking active", "training": False})


@app.route("/api/pick", methods=["POST"])
def api_pick():
    """
    Primary dynamic target pick endpoint.
    Accepts: {"target": "<object_name>", "color": "<optional_color>"}
    If a supported color is specified or detected in the target string (e.g. 'red lego brick',
    'green block', 'yellow lego'), the detection task is offloaded to vision_detection.py (30 FPS HSV).
    Otherwise starts Gemini VLM localization + CSRT tracking + Visual Servoing.
    """
    global vision_mode, active_color, active_target
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or data.get("object") or "").strip()
    color = (data.get("color") or "").strip().lower()

    if not target and not color:
        return jsonify({"ok": False, "error": "Target object name or color must be provided"}), 400

    # Automatically extract color from target description if not explicitly provided
    if not color:
        extracted = extract_color_from_string(target)
        if extracted:
            color = extracted

    if color in SUPPORTED_COLORS:
        display_target = target or f"{color} object"
        with vision_lock:
            vision_mode = "color"
            active_color = color
            active_target = display_target

        # Stop Gemini VLM queries while color vision is handling the task
        gemini_tracker.set_target(None)

        with tracker_lock:
            global tracker, tracker_found, tracker_bbox
            tracker = None
            tracker_found = False
            tracker_bbox = None

        with vs_lock:
            vs["active"] = True
            vs["target"] = display_target
            vs["pi_state"] = "ALIGN"
            vs["last_cmd"] = None
            vs["last_cmd_time"] = 0.0
            vs["centered_count"] = 0

        with state_lock:
            state["mode"] = "visual"
            state["target"] = display_target
            state["vision_mode"] = "color"
            state["active_color"] = active_color

        add_log(f"🎯 Color Visual Servo Activated for '{active_color}' ({display_target}) via vision_detection.py", "ok")
        sio.emit("mode_change", {"mode": "visual"})
        sio.emit("target_change", {"target": display_target, "color": active_color, "source": "vision_detection"})
        sio.emit("vs_update", {"pi_state": "ALIGN", "last_cmd": None, "target": display_target, "active": True})

        return jsonify({"ok": True, "active": True, "target": display_target, "color": active_color, "source": "vision_detection"})
    else:
        with vision_lock:
            vision_mode = "gemini"
            active_color = None
            active_target = target

        gemini_tracker.set_target(target)

        with vs_lock:
            vs["active"] = True
            vs["target"] = target
            vs["pi_state"] = "ALIGN"
            vs["last_cmd"] = None
            vs["last_cmd_time"] = 0.0
            vs["centered_count"] = 0

        with state_lock:
            state["mode"] = "visual"
            state["target"] = target
            state["vision_mode"] = "gemini"
            state["active_color"] = None

        add_log(f"🎯 Dynamic Target Visual Servo Activated for '{target}' via Gemini VLM", "ok")
        sio.emit("mode_change", {"mode": "visual"})
        sio.emit("target_change", {"target": target, "source": "gemini"})
        sio.emit("vs_update", {"pi_state": "ALIGN", "last_cmd": None, "target": target, "active": True})

        return jsonify({"ok": True, "active": True, "target": target, "source": "gemini"})


@app.route("/api/stop-tracking", methods=["POST"])
@app.route("/api/stop-pick", methods=["POST"])
def api_stop_pick():
    """Stops visual servoing, clears target, and halts frame processing."""
    global vision_mode, active_color, active_target
    with vision_lock:
        vision_mode = "idle"
        active_color = None
        active_target = None

    with vs_lock:
        vs["active"] = False
        vs["target"] = None
        vs["pi_state"] = "IDLE"
        vs["last_cmd"] = None
        vs["centered_count"] = 0

    with state_lock:
        state["mode"] = "manual"
        state["target"] = None
        state["vision_mode"] = "idle"
        state["active_color"] = None

    gemini_tracker.set_target(None)

    with tracker_lock:
        tracker = None
        tracker_found = False
        tracker_bbox = None

    add_log("🛑 Target Visual Servoing Stopped — frame sending halted", "info")
    sio.emit("mode_change", {"mode": "manual"})
    sio.emit("target_change", {"target": None})
    sio.emit("vs_update", {"pi_state": "IDLE", "last_cmd": None, "target": None, "active": False})
    sio.emit("yolo_pick", {"active": False})
    sio.emit("stop_camera")
    return jsonify({"ok": True, "active": False})


@app.route("/api/target", methods=["POST"])
def api_target():
    """Update target name without necessarily engaging movement. Empty or 'none' clears target."""
    global vision_mode, active_color, active_target
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or data.get("object") or "").strip()
    color = (data.get("color") or "").strip().lower()

    if not target or target.lower() in ("none", "null", "clear"):
        with vision_lock:
            vision_mode = "idle"
            active_color = None
            active_target = None
        gemini_tracker.set_target(None)
        with vs_lock:
            vs["target"] = None
        with state_lock:
            state["target"] = None
            state["vision_mode"] = "idle"
            state["active_color"] = None
        sio.emit("target_change", {"target": None})
        return jsonify({"ok": True, "target": None})

    if not color:
        color = extract_color_from_string(target)

    if color in SUPPORTED_COLORS:
        display_target = target or f"{color} object"
        with vision_lock:
            vision_mode = "color"
            active_color = color
            active_target = display_target
        gemini_tracker.set_target(None)
        with vs_lock:
            vs["target"] = display_target
        with state_lock:
            state["target"] = display_target
            state["vision_mode"] = "color"
            state["active_color"] = color
        sio.emit("target_change", {"target": display_target, "color": color, "source": "vision_detection"})
        return jsonify({"ok": True, "target": display_target, "color": color, "source": "vision_detection"})
    else:
        with vision_lock:
            vision_mode = "gemini"
            active_color = None
            active_target = target
        gemini_tracker.set_target(target)
        with vs_lock:
            vs["target"] = target
        with state_lock:
            state["target"] = target
            state["vision_mode"] = "gemini"
            state["active_color"] = None
        sio.emit("target_change", {"target": target, "source": "gemini"})
        return jsonify({"ok": True, "target": target, "source": "gemini"})


@app.route("/api/yolo-pick", methods=["POST"])
def api_yolo_pick():
    """Backward-compatible pick endpoint for dashboard buttons."""
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    target = (data.get("target") or state.get("target") or "").strip()

    if enabled:
        if not target:
            return jsonify({"ok": False, "error": "No target object specified. Please provide a target."}), 400
        gemini_tracker.set_target(target)
        with vs_lock:
            vs["active"] = True
            vs["target"] = target
            vs["pi_state"] = "ALIGN"
            vs["last_cmd"] = None
            vs["last_cmd_time"] = 0.0
        with state_lock:
            state["mode"] = "visual"
            state["target"] = target
        sio.emit("mode_change", {"mode": "visual"})
        sio.emit("target_change", {"target": target})
        sio.emit("yolo_pick", {"active": True})
        return jsonify({"ok": True, "active": True, "target": target})
    else:
        with vs_lock:
            vs["active"] = False
            vs["target"] = None
            vs["pi_state"] = "IDLE"
            vs["last_cmd"] = None
            vs["centered_count"] = 0
        with state_lock:
            state["mode"] = "manual"
            state["target"] = None
        gemini_tracker.set_target(None)
        sio.emit("mode_change", {"mode": "manual"})
        sio.emit("target_change", {"target": None})
        sio.emit("vs_update", {"active": False, "target": None, "pi_state": "IDLE", "last_cmd": None})
        sio.emit("yolo_pick", {"active": False})
        sio.emit("stop_camera")
        return jsonify({"ok": True, "active": False})


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json() or {}
    cmd = str(data.get("cmd", "")).upper().strip()
    VALID = {"LEFT", "RIGHT", "FORWARD", "BACK", "CENTERED"}
    if cmd not in VALID:
        return jsonify({"ok": False, "error": f"Unknown command: {cmd}"}), 400
    ok = send_serial(f"{cmd}:8")
    return jsonify({"ok": ok})


@app.route("/api/gripper/<action>", methods=["POST"])
def api_gripper(action):
    action = action.lower()
    if action == "open":
        ok = send_serial("S5:0")
        return jsonify({"ok": ok})
    elif action == "close":
        ok = send_serial("S5:180")
        return jsonify({"ok": ok})
    return jsonify({"ok": False, "error": "Invalid gripper action"}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Resets robot arm, deactivates visual servoing, clears target, and halts vision tracking."""
    global vision_mode, active_color, active_target
    with vision_lock:
        vision_mode = "idle"
        active_color = None
        active_target = None

    with vs_lock:
        vs["active"] = False
        vs["target"] = None
        vs["pi_state"] = "IDLE"
        vs["last_cmd"] = None
        vs["centered_count"] = 0

    with state_lock:
        state["mode"] = "manual"
        state["target"] = None
        state["vision_mode"] = "idle"
        state["active_color"] = None

    gemini_tracker.set_target(None)

    with tracker_lock:
        tracker = None
        tracker_found = False
        tracker_bbox = None

    ok = send_serial("RESET")
    add_log("Robot reset commanded — tracking and vision frames halted", "ok")
    sio.emit("reset")
    sio.emit("mode_change", {"mode": "manual"})
    sio.emit("target_change", {"target": None})
    sio.emit("vs_update", {"active": False, "target": None, "pi_state": "IDLE", "last_cmd": None})
    sio.emit("yolo_pick", {"active": False})
    sio.emit("stop_camera")
    return jsonify({"ok": ok})


@app.route("/api/servo", methods=["POST"])
def api_servo():
    data = request.get_json() or {}
    try:
        servo = int(data.get("servo"))
        angle = int(data.get("angle"))
        if not (0 <= servo <= 5 and 0 <= angle <= 180):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "Invalid servo or angle"}), 400

    cmd = f"S{servo}:{angle}"
    ok = send_serial(cmd)
    if ok:
        with state_lock:
            if servo < len(state["angles"]):
                state["angles"][servo] = angle
        sio.emit("joint_update", {"angles": state["angles"]})
    return jsonify({"ok": ok})


@app.route("/api/mode", methods=["POST"])
def api_mode():
    data = request.get_json() or {}
    mode = data.get("mode", "manual").lower()
    with state_lock:
        state["mode"] = mode
    with vs_lock:
        vs["active"] = (mode == "visual")
    sio.emit("mode_change", {"mode": mode})
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/status")
def api_status():
    with state_lock:
        s = dict(state)
    with vs_lock:
        v = dict(vs)
    return jsonify({"state": s, "vs": v})


@app.route("/api/ports")
def api_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports, "current": SERIAL_PORT})


# ── SocketIO Handlers ─────────────────────────────────────────────────────────

@sio.on("connect")
def on_connect():
    with state_lock:
        s = dict(state)
    emit("init", s)
    add_log("Dashboard client connected", "ok")


@sio.on("update_angles")
def on_update_angles(data):
    angles = data.get("angles", [])
    if len(angles) == 6:
        with state_lock:
            state["angles"] = angles
        sio.emit("joint_update", {"angles": angles})


# ── Server Boot ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"==================================================")
    print(f"🚀 Starting server_v9 on port {PORT}")
    print(f"🎯 Target Object: {gemini_tracker.target or 'None (awaiting command)'}")
    print(f"📡 Serial Port: {SERIAL_PORT} @ {BAUD_RATE}")
    print(f"⏱️ Intervals: Gemini={COMMAND_INTERVAL}s, Color Vision={COLOR_COMMAND_INTERVAL}s")
    print(f"==================================================")

    # Start background threads
    threading.Thread(target=serial_reader_thread, daemon=True).start()
    gemini_tracker.start()
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=visual_servo_thread, daemon=True).start()

    # Determine SSL context if available
    cert_file = "certs/cert.pem"
    key_file = "certs/key.pem"
    ssl_context = (cert_file, key_file) if (os.path.exists(cert_file) and os.path.exists(key_file)) else None

    if ssl_context:
        print(f"🔒 SSL enabled using {cert_file} and {key_file}")
    else:
        print("⚠️ SSL not found — running on plain HTTP")

    sio.run(
        app,
        host=HOST,
        port=PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context
    )
