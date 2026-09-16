import cv2
import json
import os
import re
import serial
import threading
import time

from flask import Flask, Response, jsonify, render_template_string
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-robotics-er-2-preview"

TARGET = "lego brick"

CAMERA_INDEX = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Gemini inference interval.
#
# IMPORTANT:
# Camera remains 30 FPS.
# Gemini only processes one frame every 0.5 seconds.
#
GEMINI_INTERVAL = 0.5

GEMINI_JPEG_QUALITY = 80
STREAM_JPEG_QUALITY = 60


# ============================================================
# SERIAL CONFIGURATION
# ============================================================

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Set False for actual ESP32 movement.
# Set True if you want to test without moving the arm.
DRY_RUN = False

ser_conn = None
serial_lock = threading.Lock()


# ============================================================
# VISUAL SERVOING CONFIGURATION
# ============================================================

CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# Dead zone
DEAD_ZONE_X = 80
DEAD_ZONE_Y = 80

# P-controller gains
KP_X = 0.03
KP_Y = 0.05

# Maximum movement command
MAX_STEP_X =  5
MAX_STEP_Y =  8

# Minimum time between ESP32 commands
COMMAND_INTERVAL = 0.5

# Minimum settling time after command
VS_SETTLE_TIME = 0.9

# Pickup area.
#
# CHANGE THIS TO YOUR ACTUAL VALUE.
PICKUP_AREA = 90000


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL CAMERA
# ============================================================

latest_camera_frame = None

camera_lock = threading.Lock()


# ============================================================
# GLOBAL VS STATE
# ============================================================

# Detection starts OFF.
#
# Gemini can still run in the background,
# but it will NOT send movement commands
# until START DETECTION is pressed.

vs_running = False

last_consumed_gemini_version = 0

last_command_time = 0.0

vs_state_lock = threading.Lock()


# ============================================================
# SERIAL CONNECTION
# ============================================================

def get_serial():

    global ser_conn

    if DRY_RUN:

        print("[SERIAL] DRY RUN enabled.")
        return None

    try:

        ser_conn = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.1
        )

        time.sleep(1.5)

        print(
            f"[SERIAL] Connected: {SERIAL_PORT}"
        )

        return ser_conn

    except serial.SerialException as e:

        print(
            f"[SERIAL] Connection failed: {e}"
        )

        ser_conn = None

        return None


# ============================================================
# SERIAL READER
# ============================================================

def serial_reader_thread():

    global ser_conn

    if DRY_RUN:

        print(
            "[SERIAL] Reader disabled because DRY_RUN=True"
        )

        return

    while True:

        if (
            ser_conn is None
            or not ser_conn.is_open
        ):

            ser_conn = get_serial()

            if ser_conn is None:

                time.sleep(3)

                continue

        try:

            if ser_conn.in_waiting:

                line = (
                    ser_conn
                    .readline()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                    .strip()
                )

                if line:

                    print(
                        f"[ESP32] {line}"
                    )

        except (
            serial.SerialException,
            OSError
        ) as e:

            print(
                f"[SERIAL] Read error: {e}"
            )

            ser_conn = None

        time.sleep(0.005)


# ============================================================
# SEND SERIAL COMMAND
# ============================================================

def send_serial(cmd):

    global ser_conn

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if DRY_RUN:

        print(
            f"[SERIAL][DRY RUN] → {cmd}"
        )

        return True

    # --------------------------------------------------------
    # Connection check
    # --------------------------------------------------------

    if (
        ser_conn is None
        or not ser_conn.is_open
    ):

        print(
            f"[SERIAL] Not connected. "
            f"Command skipped: {cmd}"
        )

        return False

    try:

        with serial_lock:

            ser_conn.write(
                (cmd + "\n").encode()
            )

            ser_conn.flush()

        print(
            f"[SERIAL] → {cmd}"
        )

        return True

    except serial.SerialException as e:

        print(
            f"[SERIAL] Send error: {e}"
        )

        ser_conn = None

        return False


# ============================================================
# GEMINI ROBOTICS
# ============================================================

class GeminiRobotics:

    def __init__(self, target="eraser"):

        api_key = os.environ.get(
            "GEMINI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.target = target

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.latest_frame = None

        self.frame_lock = threading.Lock()

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        self.latest_result = None

        self.result_lock = threading.Lock()

        # IMPORTANT:
        #
        # This increases ONLY when Gemini actually
        # produces a new result.
        #
        self.result_version = 0

        # ----------------------------------------------------
        # Worker
        # ----------------------------------------------------

        self.running = False

        self.worker = None

        self.last_request_time = 0.0

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        self.last_request_started = 0.0

        self.last_result_time = 0.0

        self.last_latency = 0.0

        self.inference_count = 0

    # ========================================================
    # UPDATE FRAME
    # ========================================================

    def update_frame(self, frame):

        with self.frame_lock:

            self.latest_frame = frame.copy()

    # ========================================================
    # GET RESULT
    # ========================================================

    def get_result(self):

        with self.result_lock:

            if self.latest_result is None:

                return None

            return self.latest_result.copy()

    # ========================================================
    # GET VERSION
    # ========================================================

    def get_result_version(self):

        with self.result_lock:

            return self.result_version

    # ========================================================
    # PROMPT
    # ========================================================

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

box_2d format:

[ymin, xmin, ymax, xmax]

IMPORTANT:

- The box must tightly surround ONLY the target.
- Do not include the table.
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

    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    def parse_json(self, text):

        if not text:

            return None

        text = text.strip()

        # Remove markdown fences

        if text.startswith("```"):

            lines = text.splitlines()

            if lines:

                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):

                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        # Direct JSON

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:

            pass

        # Extract JSON object

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:

            print(
                "[GEMINI] Could not find JSON."
            )

            return None

        try:

            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            print(
                "[GEMINI] Invalid JSON:"
            )

            print(text)

            return None

    # ========================================================
    # PROCESS RESULT
    # ========================================================

    def process_result(
        self,
        raw,
        frame
    ):

        height, width = frame.shape[:2]

        target = raw.get(
            "label",
            self.target
        )

        box = raw.get(
            "box_2d"
        )

        # ----------------------------------------------------
        # NOT FOUND
        # ----------------------------------------------------

        if box is None:

            annotated = frame.copy()

            cv2.drawMarker(
                annotated,
                (
                    CENTER_X,
                    CENTER_Y
                ),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                20,
                1
            )

            cv2.putText(
                annotated,
                f"{target}: NOT FOUND",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            return {
                "found": False,
                "centroid": None,
                "area": 0.0,
                "annotated": annotated,
                "class_id": None,
                "class_name": target,
                "confidence": 0.0
            }

        # ----------------------------------------------------
        # VALIDATE BOX
        # ----------------------------------------------------

        if (
            not isinstance(box, list)
            or len(box) != 4
        ):

            print(
                f"[GEMINI] Invalid box: {box}"
            )

            return None

        try:

            ymin, xmin, ymax, xmax = (
                float(v)
                for v in box
            )

        except (
            ValueError,
            TypeError
        ):

            return None

        # ----------------------------------------------------
        # NORMALIZED → PIXELS
        # ----------------------------------------------------

        x1 = int(
            xmin
            / 1000.0
            * width
        )

        y1 = int(
            ymin
            / 1000.0
            * height
        )

        x2 = int(
            xmax
            / 1000.0
            * width
        )

        y2 = int(
            ymax
            / 1000.0
            * height
        )

        # ----------------------------------------------------
        # CLAMP
        # ----------------------------------------------------

        x1 = max(
            0,
            min(width - 1, x1)
        )

        x2 = max(
            0,
            min(width - 1, x2)
        )

        y1 = max(
            0,
            min(height - 1, y1)
        )

        y2 = max(
            0,
            min(height - 1, y2)
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            return None

        # ----------------------------------------------------
        # CENTROID
        # ----------------------------------------------------

        cx = (
            x1 + x2
        ) // 2

        cy = (
            y1 + y2
        ) // 2

        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        area = float(
            (x2 - x1)
            * (y2 - y1)
        )

        # ----------------------------------------------------
        # ANNOTATION
        # ----------------------------------------------------

        annotated = frame.copy()

        # Camera center

        cv2.drawMarker(
            annotated,
            (
                CENTER_X,
                CENTER_Y
            ),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            1
        )

        # Bounding box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 229, 160),
            2
        )

        # Centroid

        cv2.circle(
            annotated,
            (cx, cy),
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            annotated,
            f"{target} [ER-2]",
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 229, 160),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"({cx},{cy})",
            (
                x1,
                min(
                    height - 10,
                    y2 + 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        return {
            "found": True,
            "centroid": (cx, cy),
            "area": area,
            "annotated": annotated,
            "class_id": None,
            "class_name": target,
            "confidence": 1.0
        }

    # ========================================================
    # GEMINI INFERENCE
    # ========================================================

    def detect(
        self,
        frame,
        target
    ):

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                GEMINI_JPEG_QUALITY
            ]
        )

        if not success:

            raise RuntimeError(
                "Could not encode camera frame."
            )

        prompt = self.build_prompt()

        response = (
            self.client
            .models
            .generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=encoded.tobytes(),
                        mime_type="image/jpeg"
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    thinking_config=(
                        types.ThinkingConfig(
                            thinking_level="low"
                        )
                    )
                )
            )
        )

        text = response.text

        raw = self.parse_json(
            text
        )

        return raw

    # ========================================================
    # BACKGROUND WORKER
    # ========================================================

    def worker_thread(self):

        print(
            "[GEMINI] Worker started."
        )

        while self.running:

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

            now = time.time()

            if (
                now
                - self.last_request_time
                < GEMINI_INTERVAL
            ):

                time.sleep(0.01)

                continue

            # ------------------------------------------------
            # Grab newest frame
            # ------------------------------------------------

            with self.frame_lock:

                if self.latest_frame is None:

                    time.sleep(0.05)

                    continue

                frame = (
                    self.latest_frame.copy()
                )

                target = self.target

            # ------------------------------------------------
            # Mark request
            # ------------------------------------------------

            self.last_request_time = now

            self.last_request_started = now

            try:

                print(
                    f"[GEMINI] Looking for: {target}"
                )

                raw = self.detect(
                    frame,
                    target
                )

                if raw is None:

                    print(
                        "[GEMINI] No valid response."
                    )

                    continue

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Process the SAME frame that was actually
                # sent to Gemini.
                #
                # ------------------------------------------------

                result = self.process_result(
                    raw,
                    frame
                )

                if result is None:

                    continue

                # ------------------------------------------------
                # Publish result
                # ------------------------------------------------

                with self.result_lock:

                    self.latest_result = result

                    # IMPORTANT:
                    #
                    # Version changes ONLY here.
                    #
                    self.result_version += 1

                    self.last_result_time = (
                        time.time()
                    )

                    self.last_latency = (
                        time.time()
                        - self.last_request_started
                    )

                    self.inference_count += 1

                    version = (
                        self.result_version
                    )

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                if result["found"]:

                    print(
                        f"[GEMINI] "
                        f"#{version} "
                        f"{result['class_name']} "
                        f"centroid="
                        f"{result['centroid']} "
                        f"area="
                        f"{int(result['area'])} "
                        f"latency="
                        f"{self.last_latency:.3f}s"
                    )

                else:

                    print(
                        f"[GEMINI] "
                        f"#{version} "
                        f"{target} NOT FOUND "
                        f"latency="
                        f"{self.last_latency:.3f}s"
                    )

            except Exception as e:

                print(
                    f"[GEMINI] Error: {e}"
                )

        print(
            "[GEMINI] Worker stopped."
        )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.running:

            return

        self.running = True

        self.worker = threading.Thread(
            target=self.worker_thread,
            daemon=True
        )

        self.worker.start()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False


# ============================================================
# CREATE GEMINI
# ============================================================

gemini = GeminiRobotics(
    TARGET
)


# ============================================================
# P CONTROLLER
# ============================================================

def decide_command(
    err_x,
    err_y,
    area
):

    # --------------------------------------------------------
    # Dead zones
    # --------------------------------------------------------

    in_x = (
        abs(err_x)
        <= DEAD_ZONE_X
    )

    in_y = (
        abs(err_y)
        <= DEAD_ZONE_Y
    )

    in_area = (
        area
        >= PICKUP_AREA
    )

    # --------------------------------------------------------
    # PICKUP
    # --------------------------------------------------------

    if (
        in_x
        and in_y
        and in_area
    ):

        return "CATCH"

    # --------------------------------------------------------
    # P CONTROL
    # --------------------------------------------------------

    movement_x = (
        KP_X
        * err_x
    )

    movement_y = (
        KP_Y
        * err_y
    )

    # --------------------------------------------------------
    # Limit movement
    # --------------------------------------------------------

    movement_x = max(
        -MAX_STEP_X,
        min(
            MAX_STEP_X,
            int(movement_x)
        )
    )

    movement_y = max(
        -MAX_STEP_Y,
        min(
            MAX_STEP_Y,
            int(movement_y)
        )
    )

    # --------------------------------------------------------
    # Prevent zero movement when outside dead zone
    # --------------------------------------------------------

    if movement_x == 0:

        movement_x = 1

    if movement_y == 0:

        movement_y = 1

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    if not in_x:

        if err_x > 0:

            return (
                f"RIGHT:{movement_x}"
            )

        else:

            return (
                f"LEFT:{abs(movement_x)}"
            )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    if not in_y:

        if err_y > 0:

            return (
                f"FORWARD:{movement_y}"
            )

        else:

            return (
                f"BACK:{abs(movement_y)}"
            )

    # --------------------------------------------------------
    # Centered but not yet close enough
    # --------------------------------------------------------

    return f"CENTERED:{MAX_STEP_Y-2}"


# ============================================================
# VISUAL SERVO THREAD
# ============================================================

def visual_servo_thread():

    global last_consumed_gemini_version
    global last_command_time

    print(
        "[VS] Visual servoing started."
    )

    while True:

        # ----------------------------------------------------
        # Detection state
        # ----------------------------------------------------

        with vs_state_lock:

            active = vs_running

        if not active:

            time.sleep(
                0.05
            )

            continue

        # ----------------------------------------------------
        # Check for NEW Gemini result
        # ----------------------------------------------------

        version = (
            gemini.get_result_version()
        )

        if (
            version
            <= last_consumed_gemini_version
        ):

            # No new Gemini result.
            #
            # DO NOT send anything.
            #

            time.sleep(
                0.01
            )

            continue

        # ----------------------------------------------------
        # Get result
        # ----------------------------------------------------

        result = (
            gemini.get_result()
        )

        if result is None:

            time.sleep(
                0.01
            )

            continue

        # ----------------------------------------------------
        # Consume exactly once
        # ----------------------------------------------------

        last_consumed_gemini_version = (
            version
        )

        # ----------------------------------------------------
        # Target missing
        # ----------------------------------------------------

        if not result["found"]:

            print(
                f"[VS] Gemini #{version}: "
                "target not found"
            )

            continue

        # ----------------------------------------------------
        # Calculate error
        # ----------------------------------------------------

        cx, cy = (
            result["centroid"]
        )

        err_x = (
            cx
            - CENTER_X
        )

        err_y = (
            cy
            - CENTER_Y
        )

        area = result["area"]

        print(
            f"[VS] Gemini #{version} "
            f"error=({err_x},{err_y}) "
            f"area={int(area)}"
        )

        # ----------------------------------------------------
        # Command rate limit
        # ----------------------------------------------------

        now = time.time()

        if (
            now
            - last_command_time
            < COMMAND_INTERVAL
        ):

            print(
                "[VS] Command rate limited."
            )

            continue

        # ----------------------------------------------------
        # Decide movement
        # ----------------------------------------------------

        cmd = decide_command(
            err_x,
            err_y,
            area
        )

        if cmd is None:

            continue

        # ----------------------------------------------------
        # CATCH
        # ----------------------------------------------------

        if cmd == "CATCH":

            print(
                "[VS] TARGET CENTERED."
            )

            print(
                "[VS] → CATCH"
            )

            if send_serial(
                "CATCH"
            ):

                last_command_time = (
                    time.time()
                )

            continue

        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

        print(
            f"[VS] → {cmd}"
        )

        if send_serial(
            cmd
        ):

            last_command_time = (
                time.time()
            )


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread():

    global latest_camera_frame

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    if not cap.isOpened():

        print(
            "[CAMERA] Failed to open camera."
        )

        return

    print(
        "[CAMERA] Camera opened."
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "[CAMERA] Frame read failed."
            )

            time.sleep(
                0.1
            )

            continue

        # ----------------------------------------------------
        # Store newest camera frame
        # ----------------------------------------------------

        with camera_lock:

            latest_camera_frame = (
                frame.copy()
            )

        # ----------------------------------------------------
        # Give Gemini newest frame.
        #
        # THIS NEVER WAITS FOR GEMINI.
        # ----------------------------------------------------

        gemini.update_frame(
            frame
        )


# ============================================================
# WEB STREAM
# ============================================================

def generate_frames():

    while True:

        result = (
            gemini.get_result()
        )

        # ----------------------------------------------------
        # Prefer latest Gemini annotation
        # ----------------------------------------------------

        if result is not None:

            frame = (
                result["annotated"]
                .copy()
            )

        else:

            with camera_lock:

                if (
                    latest_camera_frame
                    is None
                ):

                    time.sleep(
                        0.03
                    )

                    continue

                frame = (
                    latest_camera_frame
                    .copy()
                )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        version = (
            gemini.get_result_version()
        )

        with gemini.result_lock:

            latency = (
                gemini.last_latency
            )

            last_result_time = (
                gemini.last_result_time
            )

            inference_count = (
                gemini.inference_count
            )

        if last_result_time:

            result_age = (
                time.time()
                - last_result_time
            )

        else:

            result_age = 0.0

        with vs_state_lock:

            detecting = vs_running

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "GEMINI ER-2 PREVIEW",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 229, 160),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Result: #{version}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Latency: {latency:.2f}s",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Result age: {result_age:.2f}s",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Inferences: {inference_count}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "DETECTION: "
                + (
                    "ON"
                    if detecting
                    else "OFF"
                )
            ),
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0)
            if detecting
            else (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "SERIAL: "
                + (
                    "DRY RUN"
                    if DRY_RUN
                    else "ENABLED"
                )
            ),
            (10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                STREAM_JPEG_QUALITY
            ]
        )

        if not success:

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        time.sleep(
            0.033
        )


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Gemini ER-2 Preview Robot Test</title>

<style>

body {
    background: #111;
    color: white;
    font-family: monospace;
    text-align: center;
}

img {
    width: 640px;
    max-width: 95vw;
    border: 2px solid #444;
}

.controls {
    margin-top: 20px;

    display: flex;

    justify-content: center;

    gap: 12px;

    flex-wrap: wrap;
}

button {

    padding: 14px 28px;

    font-size: 16px;

    font-family: monospace;

    border: none;

    border-radius: 6px;

    cursor: pointer;
}

#startButton {

    background: #159447;

    color: white;
}

#stopButton {

    background: #555;

    color: white;
}

#resetButton {

    background: #c62828;

    color: white;
}

button:disabled {

    opacity: 0.45;

    cursor: not-allowed;
}

#status {

    margin-top: 15px;

    color: #aaa;
}

</style>

</head>


<body>

<h1>
Gemini ER-2 Preview + Serial VS
</h1>


<img src="/video">


<div class="controls">

    <button
        id="startButton"
        type="button"
    >
        START DETECTION
    </button>


    <button
        id="stopButton"
        type="button"
        disabled
    >
        STOP DETECTION
    </button>


    <button
        id="resetButton"
        type="button"
    >
        RESET ARM
    </button>

</div>


<div id="status">
    Detection stopped
</div>


<script>


// ==========================================================
// ELEMENTS
// ==========================================================

const startButton =
    document.getElementById(
        "startButton"
    );


const stopButton =
    document.getElementById(
        "stopButton"
    );


const resetButton =
    document.getElementById(
        "resetButton"
    );


const status =
    document.getElementById(
        "status"
    );


// ==========================================================
// START DETECTION
// ==========================================================

startButton.addEventListener(
    "click",
    async function() {

        startButton.disabled = true;

        status.textContent =
            "Starting detection...";

        try {

            const response =
                await fetch(
                    "/start-detection",
                    {
                        method: "POST"
                    }
                );

            const data =
                await response.json();

            if (data.success) {

                status.textContent =
                    "✓ Detection ACTIVE";

                stopButton.disabled =
                    false;

            } else {

                status.textContent =
                    "✗ Failed to start";

                startButton.disabled =
                    false;
            }

        } catch (error) {

            console.error(error);

            status.textContent =
                "✗ Server connection error";

            startButton.disabled =
                false;
        }
    }
);


// ==========================================================
// STOP DETECTION
// ==========================================================

stopButton.addEventListener(
    "click",
    async function() {

        stopButton.disabled = true;

        status.textContent =
            "Stopping detection...";

        try {

            const response =
                await fetch(
                    "/stop-detection",
                    {
                        method: "POST"
                    }
                );

            const data =
                await response.json();

            if (data.success) {

                status.textContent =
                    "✓ Detection stopped";

                startButton.disabled =
                    false;

            } else {

                status.textContent =
                    "✗ Failed to stop";

                stopButton.disabled =
                    false;
            }

        } catch (error) {

            console.error(error);

            status.textContent =
                "✗ Server connection error";

            stopButton.disabled =
                false;
        }
    }
);


// ==========================================================
// RESET ARM
// ==========================================================

resetButton.addEventListener(
    "click",
    async function() {

        resetButton.disabled = true;

        status.textContent =
            "Sending RESET...";

        try {

            const response =
                await fetch(
                    "/reset",
                    {
                        method: "POST"
                    }
                );

            const data =
                await response.json();

            if (data.success) {

                status.textContent =
                    "✓ RESET sent to ESP32";

            } else {

                status.textContent =
                    "✗ " + data.message;
            }

        } catch (error) {

            console.error(error);

            status.textContent =
                "✗ Server connection error";

        } finally {

            setTimeout(
                function() {

                    resetButton.disabled =
                        false;

                },
                500
            );
        }
    }
);


</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/status")
def status():

    with gemini.result_lock:

        last_result = (
            gemini.last_result_time
        )

        if last_result:

            age = (
                time.time()
                - last_result
            )

        else:

            age = None

        gemini_version = (
            gemini.result_version
        )

        gemini_count = (
            gemini.inference_count
        )

        gemini_latency = (
            gemini.last_latency
        )

    with vs_state_lock:

        detection_active = (
            vs_running
        )

    return jsonify({

        "model":
            MODEL,

        "target":
            TARGET,

        "gemini_result_version":
            gemini_version,

        "gemini_inference_count":
            gemini_count,

        "gemini_latency":
            gemini_latency,

        "gemini_result_age":
            age,

        "detection_active":
            detection_active,

        "serial_dry_run":
            DRY_RUN,

        "serial_connected":
            (
                ser_conn is not None
                and ser_conn.is_open
            )
    })


# ============================================================
# RESET ROUTE
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset_robot():

    success = send_serial(
        "RESET"
    )

    if success:

        print(
            "[WEB] RESET command sent to ESP32"
        )

        return jsonify({

            "success":
                True,

            "message":
                "RESET sent"
        })

    print(
        "[WEB] Failed to send RESET"
    )

    return jsonify({

        "success":
            False,

        "message":
            "ESP32 not connected"

    }), 503


# ============================================================
# START DETECTION ROUTE
# ============================================================

@app.route(
    "/start-detection",
    methods=["POST"]
)
def start_detection():

    global vs_running

    with vs_state_lock:

        vs_running = True

    print(
        "[WEB] Detection STARTED"
    )

    return jsonify({

        "success":
            True,

        "message":
            "Detection started"
    })


# ============================================================
# STOP DETECTION ROUTE
# ============================================================

@app.route(
    "/stop-detection",
    methods=["POST"]
)
def stop_detection():

    global vs_running

    with vs_state_lock:

        vs_running = False

    print(
        "[WEB] Detection STOPPED"
    )

    # Immediately stop/reset the arm.
    send_serial(
        "RESET"
    )

    return jsonify({

        "success":
            True,

        "message":
            "Detection stopped"
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        " Gemini ER-2 Preview + Serial VS"
    )

    print(
        "=============================================="
    )

    print(
        f"Model          : {MODEL}"
    )

    print(
        f"Target         : {TARGET}"
    )

    print(
        f"Camera         : {CAMERA_INDEX}"
    )

    print(
        f"Gemini interval: {GEMINI_INTERVAL}s"
    )

    print(
        f"Serial         : {SERIAL_PORT}"
    )

    print(
        f"Dry run        : {DRY_RUN}"
    )

    print(
        "Detection      : OFF"
    )

    print(
        "=============================================="
    )

    print()


    # --------------------------------------------------------
    # Serial
    # --------------------------------------------------------

    if not DRY_RUN:

        get_serial()

        threading.Thread(
            target=serial_reader_thread,
            daemon=True
        ).start()


    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    gemini.start()


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    threading.Thread(
        target=camera_thread,
        daemon=True
    ).start()


    # --------------------------------------------------------
    # Visual servoing
    # --------------------------------------------------------

    threading.Thread(
        target=visual_servo_thread,
        daemon=True
    ).start()


    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )