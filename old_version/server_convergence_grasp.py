import cv2
import json
import os
import re
import serial
import threading
import time

import numpy as np

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

# Gemini is ONLY used for target identification/re-acquisition.
GEMINI_INTERVAL = 1.0

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
MAX_TRACK_JUMP = 50.0

# Maximum frame-to-frame area change.
# 1.5 means area may increase up to 1.5x or decrease to 1/1.5x.
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

KP_X = 0.03
KP_Y = 0.03

MAX_STEP_X = 5
MAX_STEP_Y = 5

# Independent of Gemini.
COMMAND_INTERVAL = 0.5


# ============================================================
# POSITION-ONLY CONVERGENCE GRASPING
# ============================================================

# CATCH is based only on tracked centroid alignment.
# Area, depth, and gripper vision are not used.
GRASP_CONVERGENCE_FRAMES = 5


# ============================================================
# GRIPPER VISION
# ============================================================

# The camera view in your screenshot shows the open gripper
# in the lower portion of the image. We only search there.
GRIPPER_ROI_Y_START = 0.68
GRIPPER_ROI_Y_END = 0.99

# Bright metallic gripper-tip threshold.
# These are intentionally conservative and can be tuned from
# the dashboard without changing the tracker.
GRIPPER_V_THRESHOLD = 150
GRIPPER_S_THRESHOLD = 120

# Minimum contour size so table noise is ignored.
GRIPPER_MIN_CONTOUR_AREA = 35

# Maximum distance between a candidate and the expected
# left/right gripper regions.
GRIPPER_LEFT_MAX_X = 0.45
GRIPPER_RIGHT_MIN_X = 0.55

# Smooth the detected finger positions.
GRIPPER_SMOOTHING = 0.35

gripper_left = None
gripper_right = None
gripper_center = None
gripper_opening = None
gripper_detection_confidence = 0.0
gripper_last_update = 0.0

gripper_lock = threading.Lock()


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
# GRIPPER GEOMETRY — MOTION BASED
# ============================================================

# IMPORTANT:
# We no longer search for "bright" objects.
#
# The gripper is part of the robot and therefore moves while
# the table/background remains stationary. We exploit that:
#
#       previous frame
#              │
#              ▼
#        frame difference
#              │
#              ▼
#       moving pixels only
#              │
#        ┌─────┴─────┐
#        ▼           ▼
#     LEFT ROI    RIGHT ROI
#        │           │
#        └─────┬─────┘
#              ▼
#       gripper geometry
#
# Once a finger has been acquired, its last valid position is
# held when the arm stops. This is intentional: a static claw
# should not become "lost" merely because it stopped moving.

GRIPPER_LEFT_ROI = (
    0.00,
    0.62,
    0.45,
    0.995
)

GRIPPER_RIGHT_ROI = (
    0.55,
    0.62,
    1.00,
    0.995
)

# Frame-difference threshold.
GRIPPER_MOTION_THRESHOLD = 22

# Ignore tiny changes caused by camera noise/compression.
GRIPPER_MIN_MOTION_AREA = 80

# A moving candidate cannot jump unrealistically far.
GRIPPER_MAX_JUMP = 90.0

# Smooth accepted motion.
GRIPPER_SMOOTHING = 0.30

# Number of consecutive motion frames required before an
# uninitialized finger is accepted.
GRIPPER_ACQUIRE_STREAK = 2

# If a finger has not been seen moving for this long, we keep
# the last point rather than deleting it.
GRIPPER_HOLD_SECONDS = 2.0

# A sudden jump is rejected rather than replacing the point.
GRIPPER_BAD_LIMIT = 3

gripper_left = None
gripper_right = None
gripper_center = None
gripper_opening = None
gripper_detection_confidence = 0.0
gripper_last_update = 0.0

gripper_previous_gray = None

gripper_left_streak = 0
gripper_right_streak = 0

gripper_bad_left = 0
gripper_bad_right = 0

gripper_lock = threading.Lock()


def _roi_pixels(frame, roi):

    height, width = frame.shape[:2]

    x1 = int(width * roi[0])
    y1 = int(height * roi[1])
    x2 = int(width * roi[2])
    y2 = int(height * roi[3])

    x1 = max(
        0,
        min(width - 1, x1)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    x2 = max(
        x1 + 1,
        min(width, x2)
    )

    y2 = max(
        y1 + 1,
        min(height, y2)
    )

    return x1, y1, x2, y2


def _moving_candidate(
    gray,
    previous_gray,
    roi,
    previous_point
):

    """
    Return the best moving region in a fixed ROI.

    This deliberately uses frame-to-frame motion rather than
    brightness or color. The table can be bright, reflective,
    or dark without affecting the detector.

    Returns:
        (x, y, confidence) or None
    """

    if previous_gray is None:

        return None

    x1, y1, x2, y2 = _roi_pixels(
        gray,
        roi
    )

    current = gray[
        y1:y2,
        x1:x2
    ]

    previous = previous_gray[
        y1:y2,
        x1:x2
    ]

    difference = cv2.absdiff(
        current,
        previous
    )

    _, motion = cv2.threshold(
        difference,
        GRIPPER_MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_OPEN,
        kernel
    )

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_CLOSE,
        kernel
    )

    motion = cv2.dilate(
        motion,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    contours, _ = cv2.findContours(
        motion,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < GRIPPER_MIN_MOTION_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(
            contour
        )

        cx = (
            x1
            + bx
            + bw / 2.0
        )

        cy = (
            y1
            + by
            + bh / 2.0
        )

        # If a previous point exists, strongly prefer motion
        # near that point.
        if previous_point is not None:

            distance = float(
                np.hypot(
                    cx - previous_point[0],
                    cy - previous_point[1]
                )
            )

            if distance > GRIPPER_MAX_JUMP:
                continue

        else:

            distance = 0.0

        # Prefer larger coherent moving regions, but don't let
        # size overwhelm spatial continuity.
        score = (
            area
            / (
                1.0
                + distance
            )
        )

        candidates.append(
            (
                score,
                {
                    "x": float(cx),
                    "y": float(cy),
                    "area": float(area),
                    "w": int(bw),
                    "h": int(bh),
                    "distance": distance
                }
            )
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best = candidates[0][1]

    confidence = min(
        1.0,
        best["area"] / 1500.0
    )

    return (
        best["x"],
        best["y"],
        confidence
    )


def _smooth_point(
    raw,
    previous
):

    if previous is None:

        return (
            float(raw[0]),
            float(raw[1])
        )

    return (
        float(
            GRIPPER_SMOOTHING * raw[0]
            + (
                1.0
                - GRIPPER_SMOOTHING
            )
            * previous[0]
        ),

        float(
            GRIPPER_SMOOTHING * raw[1]
            + (
                1.0
                - GRIPPER_SMOOTHING
            )
            * previous[1]
        )
    )


def detect_gripper(frame):

    """
    Motion-based gripper geometry.

    The detector updates only when it sees plausible movement.
    When the arm stops, the last valid geometry is held.
    """

    global gripper_previous_gray

    global gripper_left
    global gripper_right
    global gripper_center
    global gripper_opening
    global gripper_detection_confidence
    global gripper_last_update

    global gripper_left_streak
    global gripper_right_streak

    global gripper_bad_left
    global gripper_bad_right

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    previous_gray = (
        gripper_previous_gray
    )

    # Always update the previous image after processing.
    gripper_previous_gray = gray

    if previous_gray is None:

        return

    with gripper_lock:

        previous_left = (
            gripper_left
        )

        previous_right = (
            gripper_right
        )

    left_candidate = _moving_candidate(
        gray,
        previous_gray,
        GRIPPER_LEFT_ROI,
        previous_left
    )

    right_candidate = _moving_candidate(
        gray,
        previous_gray,
        GRIPPER_RIGHT_ROI,
        previous_right
    )

    with gripper_lock:

        # -----------------------------------------------
        # LEFT FINGER
        # -----------------------------------------------

        if left_candidate is None:

            gripper_left_streak = 0

            if previous_left is not None:

                gripper_bad_left = min(
                    GRIPPER_BAD_LIMIT,
                    gripper_bad_left + 1
                )

        else:

            candidate = (
                left_candidate[0],
                left_candidate[1]
            )

            if previous_left is not None:

                jump = float(
                    np.hypot(
                        candidate[0]
                        - previous_left[0],

                        candidate[1]
                        - previous_left[1]
                    )
                )

                if jump > GRIPPER_MAX_JUMP:

                    gripper_bad_left += 1
                    candidate = None

            if candidate is not None:

                gripper_left_streak += 1
                gripper_bad_left = 0

                gripper_left = _smooth_point(
                    candidate,
                    previous_left
                )

        # -----------------------------------------------
        # RIGHT FINGER
        # -----------------------------------------------

        if right_candidate is None:

            gripper_right_streak = 0

            if previous_right is not None:

                gripper_bad_right = min(
                    GRIPPER_BAD_LIMIT,
                    gripper_bad_right + 1
                )

        else:

            candidate = (
                right_candidate[0],
                right_candidate[1]
            )

            if previous_right is not None:

                jump = float(
                    np.hypot(
                        candidate[0]
                        - previous_right[0],

                        candidate[1]
                        - previous_right[1]
                    )
                )

                if jump > GRIPPER_MAX_JUMP:

                    gripper_bad_right += 1
                    candidate = None

            if candidate is not None:

                gripper_right_streak += 1
                gripper_bad_right = 0

                gripper_right = _smooth_point(
                    candidate,
                    previous_right
                )

        # -----------------------------------------------
        # COMPLETE GEOMETRY
        # -----------------------------------------------

        if (
            gripper_left is not None
            and gripper_right is not None
        ):

            gripper_center = (
                (
                    gripper_left[0]
                    + gripper_right[0]
                ) / 2.0,

                (
                    gripper_left[1]
                    + gripper_right[1]
                ) / 2.0
            )

            gripper_opening = float(
                np.hypot(
                    gripper_right[0]
                    - gripper_left[0],

                    gripper_right[1]
                    - gripper_left[1]
                )
            )

            confidence_values = []

            if left_candidate is not None:

                confidence_values.append(
                    left_candidate[2]
                )

            if right_candidate is not None:

                confidence_values.append(
                    right_candidate[2]
                )

            if confidence_values:

                gripper_detection_confidence = (
                    float(
                        sum(confidence_values)
                        / len(confidence_values)
                    )
                )

                gripper_last_update = (
                    time.time()
                )

        # ------------------------------------------------
        # Log only when useful geometry exists and changed.
        # This avoids flooding the terminal at 30 FPS.
        # ------------------------------------------------

        if (
            gripper_center is not None
            and (
                left_candidate is not None
                or right_candidate is not None
            )
        ):

            print(
                f"[GRIPPER] "
                f"L=({gripper_left[0]:.0f},"
                f"{gripper_left[1]:.0f}) "
                f"R=({gripper_right[0]:.0f},"
                f"{gripper_right[1]:.0f}) "
                f"C=({gripper_center[0]:.0f},"
                f"{gripper_center[1]:.0f}) "
                f"open={gripper_opening:.1f}px "
                f"conf={gripper_detection_confidence:.2f}"
            )


def get_gripper_state():

    with gripper_lock:

        return (
            gripper_left,
            gripper_right,
            gripper_center,
            gripper_opening,
            gripper_detection_confidence,
            gripper_last_update
        )


def draw_gripper_overlay(
    frame,
    object_centroid=None
):

    (
        left,
        right,
        center,
        opening,
        confidence,
        last_update
    ) = get_gripper_state()

    height, width = frame.shape[:2]

    left_roi = _roi_pixels(
        frame,
        GRIPPER_LEFT_ROI
    )

    right_roi = _roi_pixels(
        frame,
        GRIPPER_RIGHT_ROI
    )

    # Fixed motion-search regions.
    cv2.rectangle(
        frame,
        (
            left_roi[0],
            left_roi[1]
        ),
        (
            left_roi[2],
            left_roi[3]
        ),
        (90, 90, 90),
        1
    )

    cv2.rectangle(
        frame,
        (
            right_roi[0],
            right_roi[1]
        ),
        (
            right_roi[2],
            right_roi[3]
        ),
        (90, 90, 90),
        1
    )

    cv2.putText(
        frame,
        "LEFT MOTION ROI",
        (
            left_roi[0] + 5,
            left_roi[1] + 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "RIGHT MOTION ROI",
        (
            right_roi[0] + 5,
            right_roi[1] + 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    if (
        left is None
        or right is None
        or center is None
    ):

        cv2.putText(
            frame,
            "GRIPPER GEOMETRY: WAITING FOR MOTION",
            (10, left_roi[1] + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        return

    lx, ly = (
        int(left[0]),
        int(left[1])
    )

    rx, ry = (
        int(right[0]),
        int(right[1])
    )

    gx, gy = (
        int(center[0]),
        int(center[1])
    )

    cv2.circle(
        frame,
        (lx, ly),
        8,
        (255, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        (rx, ry),
        8,
        (255, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        (gx, gy),
        7,
        (0, 255, 255),
        -1
    )

    cv2.line(
        frame,
        (lx, ly),
        (rx, ry),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "LEFT",
        (lx - 25, ly - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "RIGHT",
        (rx - 25, ry - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "GRIPPER CENTER",
        (gx + 8, gy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        (
            f"Grip width={opening:.1f}px "
            f"conf={confidence:.2f}"
        ),
        (10, left_roi[1] + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    if object_centroid is not None:

        ox, oy = (
            int(object_centroid[0]),
            int(object_centroid[1])
        )

        dx = ox - gx
        dy = oy - gy

        cv2.line(
            frame,
            (gx, gy),
            (ox, oy),
            (0, 200, 255),
            1
        )

        cv2.putText(
            frame,
            f"OBJ-GRIP dx={dx:+d} dy={dy:+d}",
            (10, left_roi[1] + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )


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

        # Exact camera snapshot used for the latest Gemini request.
        self.latest_sent_frame = None
        self.sent_frame_id = 0
        self.sent_frame_lock = threading.Lock()

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

    def set_sent_frame(self, frame, frame_id):

        with self.sent_frame_lock:
            self.latest_sent_frame = frame.copy()
            self.sent_frame_id = frame_id

    def get_sent_frame(self):

        with self.sent_frame_lock:
            if self.latest_sent_frame is None:
                return None, 0

            return (
                self.latest_sent_frame.copy(),
                self.sent_frame_id
            )

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

            # This is the exact snapshot sent to Gemini.
            with camera_lock:
                sent_frame_id = camera_frame_id

            self.set_sent_frame(
                frame,
                sent_frame_id
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
# POSITION-ONLY CONVERGENCE STATE
# ============================================================

convergence_center_count = 0
grasp_latched = False

convergence_lock = threading.Lock()


def reset_grasp_convergence():

    global convergence_center_count
    global grasp_latched

    with convergence_lock:

        convergence_center_count = 0
        grasp_latched = False


def evaluate_grasp_convergence(
    err_x,
    err_y
):

    centered = (
        abs(err_x) <= DEAD_ZONE_X
        and abs(err_y) <= DEAD_ZONE_Y
    )

    global convergence_center_count

    with convergence_lock:

        if not centered:

            convergence_center_count = 0

            return False, 0

        convergence_center_count += 1

        ready = (
            convergence_center_count
            >= GRASP_CONVERGENCE_FRAMES
        )

        return ready, convergence_center_count


# ============================================================
# P CONTROLLER
# ============================================================

def decide_command(
    err_x,
    err_y
):

    in_x = (
        abs(err_x)
        <= DEAD_ZONE_X
    )

    in_y = (
        abs(err_y)
        <= DEAD_ZONE_Y
    )

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

    if not in_x:

        if err_x > 0:

            return (
                f"RIGHT:{movement_x}"
            )

        return (
            f"LEFT:{abs(movement_x)}"
        )

    if not in_y:

        if err_y > 0:

            return (
                f"FORWARD:{movement_y}"
            )

        return (
            f"BACK:{abs(movement_y)}"
        )

    return (
        f"CENTERED:{MAX_STEP_Y - 2}"
    )


# ============================================================
# VISUAL SERVO THREAD
# ============================================================

def visual_servo_thread():

    global last_command_time
    global vs_running
    global grasp_latched

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

            reset_grasp_convergence()

            time.sleep(0.01)

            continue

        tracker_age = (
            time.time() - last_update
        )

        if tracker_age > TRACKER_MAX_AGE:

            reset_grasp_convergence()

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

        (
            grasp_ready,
            center_count
        ) = evaluate_grasp_convergence(
            err_x,
            err_y
        )

        centered_now = (
            abs(err_x) <= DEAD_ZONE_X
            and abs(err_y) <= DEAD_ZONE_Y
        )

        print(
            f"[CONVERGENCE] "
            f"centered={centered_now} "
            f"error=({err_x:+d},{err_y:+d}) "
            f"count={center_count}/"
            f"{GRASP_CONVERGENCE_FRAMES} "
            f"ready={grasp_ready}"
        )

        if grasp_ready:

            with convergence_lock:

                already_latched = grasp_latched

                if not already_latched:
                    grasp_latched = True

            if not already_latched:

                print(
                    f"[GRASP] CONVERGED "
                    f"centroid=({cx},{cy}) "
                    f"error=({err_x:+d},{err_y:+d}) "
                    f"count={center_count}/"
                    f"{GRASP_CONVERGENCE_FRAMES} "
                    f"→ CATCH"
                )

                if send_serial("CATCH"):
                    last_command_time = time.time()

                with vs_state_lock:
                    vs_running = False

                continue

        convergence_log = (
            f" convergence={center_count}/"
            f"{GRASP_CONVERGENCE_FRAMES}"
        )

        cmd = decide_command(
            err_x,
            err_y
        )

        print(
            f"[VS] TRACK "
            f"centroid=({cx},{cy}) "
            f"error=({err_x},{err_y}) "
            f"area={int(area)} "
            f"age={tracker_age:.3f}s "
            f"{convergence_log} "
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
        # Gripper vision runs locally on every camera frame.
        # It does not wait for Gemini.
        # ----------------------------------------------------

        detect_gripper(
            frame
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
# GEMINI INPUT FRAME STREAM
# ============================================================

def generate_gemini_frames():

    while True:

        frame, sent_frame_id = gemini.get_sent_frame()

        if frame is None:

            time.sleep(0.05)
            continue

        with tracker_lock:

            tracking = tracker_found
            bbox = tracker_bbox
            tracker_age = (
                time.time() - tracker_last_update
                if tracker_last_update
                else None
            )

        if tracking and bbox is not None:

            x, y, w, h = bbox

            x1 = max(0, min(FRAME_WIDTH - 1, int(x)))
            y1 = max(0, min(FRAME_HEIGHT - 1, int(y)))
            x2 = max(0, min(FRAME_WIDTH - 1, int(x + w)))
            y2 = max(0, min(FRAME_HEIGHT - 1, int(y + h)))

            cx = int(x + w / 2)
            cy = int(y + h / 2)

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
                "CURRENT CSRT TRACKER",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Tracker ({cx},{cy}) Area={int(w*h)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                1,
                cv2.LINE_AA
            )

            if tracker_age is not None:

                cv2.putText(
                    frame,
                    f"Tracker age: {tracker_age:.3f}s",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 180, 0),
                    1,
                    cv2.LINE_AA
                )

        else:

            cv2.putText(
                frame,
                "CURRENT CSRT TRACKER: LOST",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            frame,
            f"EXACT GEMINI INPUT FRAME #{sent_frame_id}",
            (10, FRAME_HEIGHT - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 229, 160),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Gemini snapshot + current tracker",
            (10, FRAME_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        time.sleep(0.03)


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
        # Gripper geometry overlay.
        # ----------------------------------------------------

        tracked_centroid = None

        if tracking and bbox is not None:

            tracked_centroid = (
                int(bbox[0] + bbox[2] / 2),
                int(bbox[1] + bbox[3] / 2)
            )

        draw_gripper_overlay(
            frame,
            tracked_centroid
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

.screens {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 24px;
    flex-wrap: wrap;
    margin: 20px auto;
    max-width: 1320px;
}

.screen {
    flex: 1 1 620px;
    max-width: 640px;
}

.screen h2 {
    font-size: 16px;
    margin: 8px 0 10px;
    color: #aaa;
}

img {
    width: 640px;
    max-width: 95vw;
    border: 2px solid #444;
    display: block;
    margin: 0 auto;
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

<div class="screens">

    <div class="screen">

        <h2>LIVE CAMERA + TRACKER + GRIPPER MOTION</h2>

        <img src="/video">

    </div>

    <div class="screen">

        <h2>LATEST FRAME SENT TO GEMINI</h2>

        <img src="/gemini-video">

    </div>

</div>


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


@app.route("/gemini-video")
def gemini_video():

    return Response(
        generate_gemini_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/status")
def status():

    with gemini.sent_frame_lock:
        gemini_sent_frame_id = gemini.sent_frame_id

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

    with convergence_lock:

        status_center_count = convergence_center_count
        status_grasp_latched = grasp_latched

    (
        status_gripper_left,
        status_gripper_right,
        status_gripper_center,
        status_gripper_opening,
        status_gripper_confidence,
        status_gripper_last_update
    ) = get_gripper_state()

    return jsonify({

        "model":
            MODEL,

        "target":
            TARGET,

        "camera_frame_id":
            current_camera_frame,

        "gemini_sent_frame_id":
            gemini_sent_frame_id,

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
            ),

        "gripper_left":
            status_gripper_left,

        "gripper_right":
            status_gripper_right,

        "gripper_center":
            status_gripper_center,

        "gripper_opening_px":
            status_gripper_opening,

        "gripper_confidence":
            status_gripper_confidence,

        "gripper_age":
            (
                time.time()
                - status_gripper_last_update
                if status_gripper_last_update
                else None
            ),

        "grasp_convergence_frames":
            status_center_count,

        "grasp_convergence_target":
            GRASP_CONVERGENCE_FRAMES,

        "grasp_latched":
            status_grasp_latched
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

    reset_grasp_convergence()

    reset_grasp_convergence()

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

    reset_grasp_convergence()

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
        f"Gripper ROI      : "
        f"{GRIPPER_ROI_Y_START:.2f}-"
        f"{GRIPPER_ROI_Y_END:.2f}"
    )

    print(
        f"Gripper motion  : "
        f"threshold={GRIPPER_MOTION_THRESHOLD}"
    )

    print(
        f"Gripper jump    : "
        f"{GRIPPER_MAX_JUMP}px"
    )

    print(
        f"Grasp convergence: "
        f"{GRASP_CONVERGENCE_FRAMES} centered evaluations"
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
