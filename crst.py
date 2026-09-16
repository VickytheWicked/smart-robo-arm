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
TARGET = "toy brick"

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Gemini is ONLY used for target identification/re-acquisition.
GEMINI_INTERVAL = 1.5

GEMINI_JPEG_QUALITY = 80
STREAM_JPEG_QUALITY = 60


# ============================================================
# TRACKER
# ============================================================

TRACKER_TYPE = "CSRT"

# Movement commands are blocked if tracker data becomes stale.
TRACKER_MAX_AGE = 0.5


# ------------------------------------------------------------
# Anti-drift validation
# ------------------------------------------------------------

# Maximum centroid movement allowed between accepted frames.
MAX_TRACK_JUMP = 45.0

# Maximum frame-to-frame area change.
MAX_AREA_RATIO = 2.5

# Maximum aspect-ratio change between accepted frames.
MAX_ASPECT_RATIO_CHANGE = 1.35

# Consecutive rejected tracker frames before declaring loss.
TRACK_BAD_FRAME_LIMIT = 3


# ============================================================
# SERIAL
# ============================================================

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# False = send to physical ESP32.
# True = print commands only.
DRY_RUN = False

ser_conn = None
serial_lock = threading.Lock()


# ============================================================
# VISUAL SERVO
# ============================================================

CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

DEAD_ZONE_X = 80
DEAD_ZONE_Y = 80

KP_X = 0.035
KP_Y = 0.035

MAX_STEP_X = 5
MAX_STEP_Y = 5

# Independent of Gemini.
COMMAND_INTERVAL = 0.5


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CAMERA STATE
# ============================================================

latest_camera_frame = None
camera_frame_id = 0

camera_lock = threading.Lock()


# ============================================================
# TRACKER STATE
# ============================================================

tracker = None
tracker_bbox = None
tracker_found = False

tracker_last_update = 0.0
tracker_frame_id = 0

# Last accepted / known-good bbox.
tracker_previous_bbox = None

# Number of consecutive suspicious updates.
tracker_bad_frame_count = 0

# Gemini result version used for current tracker.
tracker_last_gemini_version = 0

tracker_lock = threading.Lock()


# ============================================================
# CREATE TRACKER
# ============================================================

def create_tracker():

    if hasattr(cv2, "TrackerCSRT_create"):

        return cv2.TrackerCSRT_create()

    if (
        hasattr(cv2, "legacy")
        and hasattr(
            cv2.legacy,
            "TrackerCSRT_create"
        )
    ):

        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is unavailable. "
        "Install an OpenCV build with tracker support."
    )


# ============================================================
# VISUAL SERVO STATE
# ============================================================

vs_running = False
last_command_time = 0.0

vs_state_lock = threading.Lock()


# ============================================================
# SERIAL CONNECTION
# ============================================================

def get_serial():

    global ser_conn

    if DRY_RUN:

        print(
            "[SERIAL] DRY RUN enabled. "
            "ESP32 commands will NOT be sent."
        )

        return None

    try:

        ser_conn = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.1
        )

        time.sleep(1.5)

        print(
            f"[SERIAL] Connected: "
            f"{SERIAL_PORT} @ {BAUD_RATE}"
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
# SEND SERIAL
# ============================================================

def send_serial(cmd):

    global ser_conn

    if DRY_RUN:

        print(
            f"[SERIAL DRY RUN - NOT SENT] → {cmd}"
        )

        return True

    if (
        ser_conn is None
        or not ser_conn.is_open
    ):

        print(
            f"[SERIAL] NOT CONNECTED - "
            f"command skipped: {cmd}"
        )

        return False

    try:

        with serial_lock:

            ser_conn.write(
                (cmd + "\n").encode()
            )

            ser_conn.flush()

        print(
            f"[SERIAL SENT] "
            f"{SERIAL_PORT} @ {BAUD_RATE} "
            f"→ {cmd}"
        )

        return True

    except (
        serial.SerialException,
        OSError
    ) as e:

        print(
            f"[SERIAL] Send error: {e}"
        )

        ser_conn = None

        return False


# ============================================================
# GEMINI
# ============================================================

class GeminiRobotics:

    def __init__(self, target):

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

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_result = None
        self.result_lock = threading.Lock()

        self.result_version = 0

        self.running = False
        self.worker = None

        self.last_request_time = 0.0
        self.last_request_started = 0.0

        self.last_result_time = 0.0
        self.last_latency = 0.0

        self.inference_count = 0


    def update_frame(self, frame):

        with self.frame_lock:

            self.latest_frame = frame.copy()


    def get_result(self):

        with self.result_lock:

            if self.latest_result is None:

                return None

            return self.latest_result.copy()


    def get_result_version(self):

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


    def parse_json(self, text):

        if not text:

            return None

        text = text.strip()

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

        try:

            return json.loads(text)

        except json.JSONDecodeError:

            pass

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


    def process_result(self, raw, frame):

        height, width = frame.shape[:2]

        target = raw.get(
            "label",
            self.target
        )

        box = raw.get(
            "box_2d"
        )

        if box is None:

            return {
                "found": False,
                "centroid": None,
                "area": 0.0,
                "bbox": None,
                "class_id": None,
                "class_name": target,
                "confidence": 0.0
            }

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

        x1 = int(
            xmin / 1000.0 * width
        )

        y1 = int(
            ymin / 1000.0 * height
        )

        x2 = int(
            xmax / 1000.0 * width
        )

        y2 = int(
            ymax / 1000.0 * height
        )

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

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        area = float(
            (x2 - x1)
            * (y2 - y1)
        )

        return {
            "found": True,
            "centroid": (cx, cy),
            "area": area,
            "bbox": (
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ),
            "class_id": None,
            "class_name": target,
            "confidence": 1.0
        }


    def detect(self, frame):

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
                    self.build_prompt()
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

        return self.parse_json(
            response.text
        )


    def worker_thread(self):

        print(
            "[GEMINI] Worker started."
        )

        while self.running:

            now = time.time()

            if (
                now - self.last_request_time
                < GEMINI_INTERVAL
            ):

                time.sleep(0.01)

                continue

            with self.frame_lock:

                if self.latest_frame is None:

                    time.sleep(0.05)

                    continue

                frame = (
                    self.latest_frame.copy()
                )

            self.last_request_time = now
            self.last_request_started = now

            try:

                print(
                    f"[GEMINI] Looking for: "
                    f"{self.target}"
                )

                raw = self.detect(frame)

                if raw is None:

                    print(
                        "[GEMINI] No valid response."
                    )

                    continue

                result = self.process_result(
                    raw,
                    frame
                )

                if result is None:

                    continue

                latency = (
                    time.time()
                    - self.last_request_started
                )

                with self.result_lock:

                    self.latest_result = result
                    self.result_version += 1

                    self.last_result_time = (
                        time.time()
                    )

                    self.last_latency = latency
                    self.inference_count += 1

                    version = self.result_version

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
                        f"{latency:.3f}s"
                    )

                else:

                    print(
                        f"[GEMINI] "
                        f"#{version} "
                        f"{self.target} NOT FOUND "
                        f"latency="
                        f"{latency:.3f}s"
                    )

            except Exception as e:

                print(
                    f"[GEMINI] Error: {e}"
                )

        print(
            "[GEMINI] Worker stopped."
        )


    def start(self):

        if self.running:

            return

        self.running = True

        self.worker = threading.Thread(
            target=self.worker_thread,
            daemon=True
        )

        self.worker.start()


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

    in_x = (
        abs(err_x)
        <= DEAD_ZONE_X
    )

    in_y = (
        abs(err_y)
        <= DEAD_ZONE_Y
    )


    # ========================================================
    # IMPORTANT:
    #
    # There is NO area-based CATCH anymore.
    #
    # When the object is centered, we send CENTERED.
    #
    # ESP32 is responsible for:
    #
    #     CENTERED
    #          ↓
    #     moveDown()
    #          ↓
    #     calibrated table position
    #          ↓
    #        GRAB
    #
    # ========================================================


    movement_x = int(
        KP_X * err_x
    )

    movement_y = int(
        KP_Y * err_y
    )


    movement_x = max(
        -MAX_STEP_X,
        min(
            MAX_STEP_X,
            movement_x
        )
    )


    movement_y = max(
        -MAX_STEP_Y,
        min(
            MAX_STEP_Y,
            movement_y
        )
    )


    if movement_x == 0:

        movement_x = 1


    if movement_y == 0:

        movement_y = 1


    # --------------------------------------------------------
    # X axis alignment
    # --------------------------------------------------------

    if not in_x:

        if err_x > 0:

            return (
                f"RIGHT:{movement_x-1}"
            )

        return (
            f"LEFT:{abs(movement_x)-1}"
        )


    # --------------------------------------------------------
    # Y axis alignment
    # --------------------------------------------------------

    if not in_y:

        if err_y > 0:

            return (
                f"FORWARD:{movement_y}"
            )

        return (
            f"BACK:{abs(movement_y)}"
        )


    # --------------------------------------------------------
    # Object centered
    #
    # IMPORTANT:
    #
    # Never send CATCH based on area.
    #
    # CENTERED tells ESP32 to descend.
    # --------------------------------------------------------

    return (
        f"CENTERED:{MAX_STEP_Y - 2}"
    )


# ============================================================
# VISUAL SERVO THREAD
# ============================================================

def visual_servo_thread():

    global last_command_time

    print(
        "[VS] Visual servoing started."
    )

    while True:

        with vs_state_lock:

            active = vs_running

        if not active:

            time.sleep(0.05)

            continue

        with tracker_lock:

            active_tracking = tracker_found
            bbox = tracker_bbox
            last_update = tracker_last_update

        if (
            not active_tracking
            or bbox is None
        ):

            time.sleep(0.01)

            continue

        tracker_age = (
            time.time() - last_update
        )

        if tracker_age > TRACKER_MAX_AGE:

            time.sleep(0.01)

            continue

        x, y, w, h = bbox

        cx = int(
            x + w / 2
        )

        cy = int(
            y + h / 2
        )

        area = float(
            w * h
        )

        err_x = (
            cx - CENTER_X
        )

        err_y = (
            cy - CENTER_Y
        )

        now = time.time()

        if (
            now - last_command_time
            < COMMAND_INTERVAL
        ):

            time.sleep(0.005)

            continue

        cmd = decide_command(
            err_x,
            err_y,
            area
        )


        print(
            f"[VS] TRACK "
            f"centroid=({cx},{cy}) "
            f"error=({err_x},{err_y}) "
            f"area={int(area)} "
            f"age={tracker_age:.3f}s "
            f"→ {cmd}"
        )


        if send_serial(cmd):

            last_command_time = time.time()


# ============================================================
# CAMERA + TRACKER THREAD
# ============================================================

def camera_thread():

    global latest_camera_frame
    global camera_frame_id

    global tracker
    global tracker_bbox
    global tracker_found
    global tracker_last_update
    global tracker_frame_id

    global tracker_previous_bbox
    global tracker_bad_frame_count
    global tracker_last_gemini_version


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


    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:

        pass


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

            time.sleep(0.05)

            continue


        # ----------------------------------------------------
        # Publish newest frame only.
        # ----------------------------------------------------

        with camera_lock:

            camera_frame_id += 1

            current_frame_id = (
                camera_frame_id
            )

            latest_camera_frame = (
                frame.copy()
            )


        # ----------------------------------------------------
        # Gemini receives newest frame asynchronously.
        # ----------------------------------------------------

        gemini.update_frame(
            frame
        )


        # ----------------------------------------------------
        # Gemini re-acquisition.
        # ----------------------------------------------------

        gemini_version = (
            gemini.get_result_version()
        )


        if (
            gemini_version
            > tracker_last_gemini_version
        ):

            gemini_result = (
                gemini.get_result()
            )


            if (
                gemini_result is not None
                and gemini_result["found"]
                and gemini_result.get("bbox")
                is not None
            ):

                new_bbox = tuple(
                    int(v)
                    for v
                    in gemini_result["bbox"]
                )


                try:

                    new_tracker = (
                        create_tracker()
                    )


                    new_tracker.init(
                        frame,
                        new_bbox
                    )


                    with tracker_lock:

                        tracker = new_tracker
                        tracker_bbox = new_bbox
                        tracker_found = True

                        tracker_last_update = (
                            time.time()
                        )

                        tracker_frame_id = (
                            current_frame_id
                        )

                        tracker_previous_bbox = (
                            new_bbox
                        )

                        tracker_bad_frame_count = 0


                    print(
                        f"[TRACKER] Initialized "
                        f"from Gemini #{gemini_version} "
                        f"bbox={new_bbox}"
                    )


                except Exception as e:

                    with tracker_lock:

                        tracker = None
                        tracker_bbox = None
                        tracker_found = False
                        tracker_previous_bbox = None
                        tracker_bad_frame_count = 0


                    print(
                        f"[TRACKER] Init failed: {e}"
                    )


            else:

                with tracker_lock:

                    tracker = None
                    tracker_bbox = None
                    tracker_found = False
                    tracker_previous_bbox = None
                    tracker_bad_frame_count = 0


                print(
                    f"[TRACKER] Gemini #{gemini_version}: "
                    "target not found."
                )


            tracker_last_gemini_version = (
                gemini_version
            )


        # ----------------------------------------------------
        # Fast CSRT update.
        # ----------------------------------------------------

        with tracker_lock:

            active_tracker = tracker
            active_tracking = tracker_found


        if (
            active_tracker is not None
            and active_tracking
        ):

            try:

                success, updated_bbox = (
                    active_tracker.update(frame)
                )


                if not success:

                    with tracker_lock:

                        tracker_bad_frame_count += 1

                        bad_count = (
                            tracker_bad_frame_count
                        )


                    print(
                        f"[TRACKER] Update failed "
                        f"bad={bad_count}/"
                        f"{TRACK_BAD_FRAME_LIMIT}"
                    )


                    if (
                        bad_count
                        >= TRACK_BAD_FRAME_LIMIT
                    ):

                        with tracker_lock:

                            tracker_found = False
                            tracker_bbox = None
                            tracker_previous_bbox = None
                            tracker_bad_frame_count = 0


                        print(
                            "[TRACKER] LOST. "
                            "Waiting for Gemini re-acquisition."
                        )


                else:

                    x, y, w, h = (
                        updated_bbox
                    )


                    candidate_bbox = (
                        float(x),
                        float(y),
                        float(w),
                        float(h)
                    )


                    # ----------------------------------------
                    # Basic geometry validation.
                    # ----------------------------------------

                    valid_geometry = (
                        w > 2
                        and h > 2
                        and x < FRAME_WIDTH
                        and y < FRAME_HEIGHT
                        and x + w > 0
                        and y + h > 0
                    )


                    if not valid_geometry:

                        with tracker_lock:

                            tracker_bad_frame_count += 1

                            bad_count = (
                                tracker_bad_frame_count
                            )


                        print(
                            f"[TRACKER] Invalid geometry "
                            f"bad={bad_count}/"
                            f"{TRACK_BAD_FRAME_LIMIT}"
                        )


                    else:

                        valid_update, reason = (
                            validate_tracker_update(
                                candidate_bbox
                            )
                        )


                        if valid_update:

                            with tracker_lock:

                                tracker_bbox = (
                                    candidate_bbox
                                )

                                tracker_previous_bbox = (
                                    candidate_bbox
                                )

                                tracker_found = True

                                tracker_last_update = (
                                    time.time()
                                )

                                tracker_frame_id = (
                                    current_frame_id
                                )

                                tracker_bad_frame_count = 0


                        else:

                            with tracker_lock:

                                tracker_bad_frame_count += 1

                                bad_count = (
                                    tracker_bad_frame_count
                                )


                            print(
                                f"[TRACKER] REJECTED "
                                f"{reason} "
                                f"bad={bad_count}/"
                                f"{TRACK_BAD_FRAME_LIMIT}"
                            )


                    # ----------------------------------------
                    # Repeated invalid updates -> LOST.
                    # ----------------------------------------

                    with tracker_lock:

                        bad_count = (
                            tracker_bad_frame_count
                        )


                    if (
                        bad_count
                        >= TRACK_BAD_FRAME_LIMIT
                    ):

                        with tracker_lock:

                            tracker_found = False
                            tracker_bbox = None
                            tracker_previous_bbox = None
                            tracker_bad_frame_count = 0


                        print(
                            "[TRACKER] LOST after "
                            "repeated suspicious updates. "
                            "Waiting for Gemini."
                        )


            except Exception as e:

                with tracker_lock:

                    tracker_found = False
                    tracker_bbox = None
                    tracker_previous_bbox = None
                    tracker_bad_frame_count = 0


                print(
                    f"[TRACKER] Update error: {e}"
                )


# ============================================================
# TRACKER VALIDATION
# ============================================================

def validate_tracker_update(new_bbox):

    with tracker_lock:

        previous = tracker_previous_bbox


    if previous is None:

        return True, "FIRST"


    px, py, pw, ph = previous
    nx, ny, nw, nh = new_bbox


    # --------------------------------------------------------
    # Centroid jump.
    # --------------------------------------------------------

    previous_cx = (
        px + pw / 2.0
    )

    previous_cy = (
        py + ph / 2.0
    )


    new_cx = (
        nx + nw / 2.0
    )

    new_cy = (
        ny + nh / 2.0
    )


    jump = (
        (new_cx - previous_cx) ** 2
        + (new_cy - previous_cy) ** 2
    ) ** 0.5


    if jump > MAX_TRACK_JUMP:

        return (
            False,
            f"CENTROID_JUMP={jump:.1f}px"
        )


    # --------------------------------------------------------
    # Area.
    # --------------------------------------------------------

    previous_area = max(
        1.0,
        pw * ph
    )


    new_area = max(
        1.0,
        nw * nh
    )


    area_ratio = (
        new_area / previous_area
    )


    if (
        area_ratio > MAX_AREA_RATIO
        or area_ratio
        < 1.0 / MAX_AREA_RATIO
    ):

        return (
            False,
            f"AREA_JUMP={area_ratio:.2f}x"
        )


    # --------------------------------------------------------
    # Aspect ratio.
    # --------------------------------------------------------

    previous_aspect = (
        pw / max(ph, 1.0)
    )


    new_aspect = (
        nw / max(nh, 1.0)
    )


    aspect_ratio = (
        new_aspect
        / max(previous_aspect, 0.001)
    )


    if (
        aspect_ratio > MAX_ASPECT_RATIO_CHANGE
        or aspect_ratio
        < 1.0 / MAX_ASPECT_RATIO_CHANGE
    ):

        return (
            False,
            f"ASPECT_JUMP={aspect_ratio:.2f}x"
        )


    return True, "OK"


# ============================================================
# WEB STREAM
# ============================================================

def generate_frames():

    while True:

        # ----------------------------------------------------
        # ALWAYS use newest camera frame.
        # Gemini never controls the displayed frame.
        # ----------------------------------------------------

        with camera_lock:

            if latest_camera_frame is None:

                time.sleep(0.02)

                continue

            frame = (
                latest_camera_frame.copy()
            )

            current_frame_id = (
                camera_frame_id
            )


        # ----------------------------------------------------
        # Current tracker.
        # ----------------------------------------------------

        with tracker_lock:

            tracking = tracker_found
            bbox = tracker_bbox

            tracker_age = (
                time.time()
                - tracker_last_update
                if tracker_last_update
                else None
            )

            tracked_frame_id = (
                tracker_frame_id
            )


        if tracking and bbox is not None:

            x, y, w, h = bbox


            x1 = max(
                0,
                min(
                    FRAME_WIDTH - 1,
                    int(x)
                )
            )


            y1 = max(
                0,
                min(
                    FRAME_HEIGHT - 1,
                    int(y)
                )
            )


            x2 = max(
                0,
                min(
                    FRAME_WIDTH - 1,
                    int(x + w)
                )
            )


            y2 = max(
                0,
                min(
                    FRAME_HEIGHT - 1,
                    int(y + h)
                )
            )


            cx = int(
                x + w / 2
            )

            cy = int(
                y + h / 2
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                2
            )


            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 180, 0),
                -1
            )


            cv2.putText(
                frame,
                "CSRT TRACKER",
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                f"Track ({cx},{cy}) "
                f"Area={int(w*h)}",
                (
                    10,
                    225
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                1,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # Gemini stats.
        # ----------------------------------------------------

        with gemini.result_lock:

            gemini_version = (
                gemini.result_version
            )

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
        # Overlay.
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "GEMINI + CSRT VISUAL SERVO",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 229, 160),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Camera frame: #{current_frame_id}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Gemini: #{gemini_version} "
            f"Latency={latency:.2f}s",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Gemini age: {result_age:.2f}s",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            (
                "TRACKER: "
                + (
                    f"ACTIVE age={tracker_age:.3f}s "
                    f"frame=#{tracked_frame_id}"
                    if tracking
                    and tracker_age is not None
                    else "LOST"
                )
            ),
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0)
            if tracking
            else (0, 0, 255),
            1,
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
            0.5,
            (0, 255, 0)
            if detecting
            else (0, 0, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"ESP32 interval: "
            f"{COMMAND_INTERVAL:.2f}s",
            (10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
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
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


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


        # Small sleep prevents the Flask generator from
        # consuming CPU unnecessarily.

        time.sleep(0.01)


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!doctype html>

<html>

<head>

<title>Gemini + CSRT Robot Arm</title>

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
Gemini + CSRT Visual Servo
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

        age = (
            time.time() - last_result
            if last_result
            else None
        )

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


    with tracker_lock:

        tracker_active = (
            tracker_found
        )

        tracker_age = (
            time.time()
            - tracker_last_update
            if tracker_last_update
            else None
        )

        current_tracker_bbox = (
            tracker_bbox
        )

        bad_frames = (
            tracker_bad_frame_count
        )


    with camera_lock:

        current_camera_frame = (
            camera_frame_id
        )


    return jsonify({

        "model":
            MODEL,

        "target":
            TARGET,

        "camera_frame_id":
            current_camera_frame,

        "gemini_result_version":
            gemini_version,

        "gemini_inference_count":
            gemini_count,

        "gemini_latency":
            gemini_latency,

        "gemini_result_age":
            age,

        "tracker_type":
            TRACKER_TYPE,

        "tracker_active":
            tracker_active,

        "tracker_age":
            tracker_age,

        "tracker_bbox":
            current_tracker_bbox,

        "tracker_bad_frame_count":
            bad_frames,

        "detection_active":
            detection_active,

        "command_interval":
            COMMAND_INTERVAL,

        "serial_dry_run":
            DRY_RUN,

        "serial_connected":
            (
                ser_conn is not None
                and ser_conn.is_open
            )
    })


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
            "success": True,
            "message": "RESET sent"
        })


    print(
        "[WEB] Failed to send RESET"
    )


    return jsonify({
        "success": False,
        "message": "ESP32 not connected"
    }), 503


@app.route(
    "/start-detection",
    methods=["POST"]
)
def start_detection():

    global vs_running
    global last_command_time


    with vs_state_lock:

        vs_running = True
        last_command_time = 0.0


    print(
        "[WEB] Detection STARTED"
    )


    return jsonify({
        "success": True,
        "message": "Detection started"
    })


@app.route(
    "/stop-detection",
    methods=["POST"]
)
def stop_detection():

    global vs_running

    global tracker
    global tracker_bbox
    global tracker_found
    global tracker_previous_bbox
    global tracker_bad_frame_count


    with vs_state_lock:

        vs_running = False


    with tracker_lock:

        tracker = None
        tracker_bbox = None
        tracker_found = False
        tracker_previous_bbox = None
        tracker_bad_frame_count = 0


    print(
        "[WEB] Detection STOPPED"
    )


    send_serial(
        "RESET"
    )


    return jsonify({
        "success": True,
        "message": "Detection stopped"
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "================================================"
    )

    print(
        " Gemini + CSRT Visual Servo Robot Arm"
    )

    print(
        "================================================"
    )

    print(
        f"Model            : {MODEL}"
    )

    print(
        f"Target           : {TARGET}"
    )

    print(
        f"Camera           : {CAMERA_INDEX}"
    )

    print(
        f"Camera FPS       : {CAMERA_FPS}"
    )

    print(
        f"Gemini interval  : {GEMINI_INTERVAL}s"
    )

    print(
        f"Tracker          : {TRACKER_TYPE}"
    )

    print(
        f"Tracker max age  : {TRACKER_MAX_AGE}s"
    )

    print(
        f"Max track jump   : {MAX_TRACK_JUMP}px"
    )

    print(
        f"Max area ratio   : {MAX_AREA_RATIO}x"
    )

    print(
        f"Max aspect ratio : "
        f"{MAX_ASPECT_RATIO_CHANGE}x"
    )

    print(
        f"Bad-frame limit  : "
        f"{TRACK_BAD_FRAME_LIMIT}"
    )

    print(
        f"ESP32 interval   : "
        f"{COMMAND_INTERVAL}s"
    )

    print(
        f"Serial            : "
        f"{SERIAL_PORT}"
    )

    print(
        f"Baud              : "
        f"{BAUD_RATE}"
    )

    print(
        f"Dry run           : "
        f"{DRY_RUN}"
    )

    print(
        "Detection         : OFF"
    )

    print(
        "================================================"
    )

    print()


    if not DRY_RUN:

        get_serial()


        threading.Thread(
            target=serial_reader_thread,
            daemon=True
        ).start()


    gemini.start()


    threading.Thread(
        target=camera_thread,
        daemon=True
    ).start()


    threading.Thread(
        target=visual_servo_thread,
        daemon=True
    ).start()


    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )