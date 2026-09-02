import cv2
import json
import os
import threading
import time

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-robotics-er-2-preview"

# Change this to whatever object you want the robot to pick.
TARGET_OBJECT = "eraser"

# How often Gemini is allowed to analyze a new frame.
# Do NOT make this extremely small yet.
GEMINI_INTERVAL = 1.5

# Camera
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Your visual-servo dead zones
DEAD_ZONE_X = 50
DEAD_ZONE_Y = 50


# ============================================================
# GEMINI CLIENT
# ============================================================

if "GEMINI_API_KEY" not in os.environ:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ============================================================
# SHARED STATE
# ============================================================

latest_frame = None

latest_result = None

gemini_busy = False

state_lock = threading.Lock()


# ============================================================
# GEMINI DETECTION
# ============================================================

def detect_target(frame, target_object):

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        raise RuntimeError("Could not encode camera frame.")

    image_bytes = encoded.tobytes()

    prompt = f"""
You are the vision system for a robotic arm.

The robot needs to pick:

"{target_object}"

Locate ONLY this target object.

Ignore all other objects.

Return ONLY valid JSON in exactly this format:

{{
    "label": "{target_object}",
    "box_2d": [ymin, xmin, ymax, xmax]
}}

Coordinate rules:

- Coordinates are normalized from 0 to 1000.
- box_2d = [top, left, bottom, right]
- ymin = top of the object
- xmin = left side of the object
- ymax = bottom of the object
- xmax = right side of the object

IMPORTANT:

- The bounding box must tightly surround ONLY the target object.
- Do not include the table or background.
- Do not include nearby objects.
- Do not merge objects.
- Do not invent the target if it is not visible.
- If the target is not visible, return:

{{
    "label": "{target_object}",
    "box_2d": null
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            )
        ),
    )

    text = response.text.strip()

    # Remove markdown fences if Gemini adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:

        print("\nInvalid JSON returned by Gemini:")
        print(text)

        raise RuntimeError(
            "Gemini returned invalid JSON."
        )

    return result


# ============================================================
# CONVERT GEMINI RESULT TO ROBOT COORDINATES
# ============================================================

def process_result(result, frame_shape):

    height, width = frame_shape[:2]

    box = result.get("box_2d")

    if box is None:

        return {
            "found": False,
            "centroid": None,
            "area": 0.0,
            "class_name": result.get(
                "label",
                TARGET_OBJECT
            ),
            "confidence": 0.0,
            "box_pixels": None,
            "err_x": 0,
            "err_y": 0
        }

    if not isinstance(box, list) or len(box) != 4:

        raise RuntimeError(
            f"Invalid box_2d returned by Gemini: {box}"
        )

    # Gemini format:
    #
    # [ymin, xmin, ymax, xmax]
    #

    ymin, xmin, ymax, xmax = box

    # Convert 0-1000 → actual image pixels.

    x1 = int(float(xmin) / 1000 * width)
    y1 = int(float(ymin) / 1000 * height)

    x2 = int(float(xmax) / 1000 * width)
    y2 = int(float(ymax) / 1000 * height)

    # Clamp to image boundaries.

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))

    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1 or y2 <= y1:

        return {
            "found": False,
            "centroid": None,
            "area": 0.0,
            "class_name": result.get(
                "label",
                TARGET_OBJECT
            ),
            "confidence": 0.0,
            "box_pixels": None,
            "err_x": 0,
            "err_y": 0
        }

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # --------------------------------------------------------
    # Bounding-box area
    # --------------------------------------------------------

    area = float(
        (x2 - x1) * (y2 - y1)
    )

    # --------------------------------------------------------
    # Image center
    # --------------------------------------------------------

    image_cx = width // 2
    image_cy = height // 2

    # --------------------------------------------------------
    # Pixel error
    #
    # Exactly the type of error your current controller uses.
    # --------------------------------------------------------

    err_x = cx - image_cx
    err_y = cy - image_cy

    return {
        "found": True,

        "centroid": (
            cx,
            cy
        ),

        "area": area,

        "class_name": result.get(
            "label",
            TARGET_OBJECT
        ),

        "confidence": 1.0,

        "box_pixels": (
            x1,
            y1,
            x2,
            y2
        ),

        "err_x": err_x,
        "err_y": err_y
    }


# ============================================================
# GEMINI WORKER
# ============================================================

def gemini_worker():

    global latest_result
    global gemini_busy

    last_request_time = 0

    while True:

        time.sleep(0.05)

        # ----------------------------------------------------
        # Don't run Gemini too frequently.
        # ----------------------------------------------------

        now = time.time()

        if now - last_request_time < GEMINI_INTERVAL:
            continue

        # ----------------------------------------------------
        # Don't start another request while one is running.
        # ----------------------------------------------------

        with state_lock:

            if gemini_busy:
                continue

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

            gemini_busy = True

        last_request_time = now

        try:

            print()
            print("----------------------------------------")
            print(
                f"Gemini → Looking for: {TARGET_OBJECT}"
            )
            print("----------------------------------------")

            raw_result = detect_target(
                frame,
                TARGET_OBJECT
            )

            processed = process_result(
                raw_result,
                frame.shape
            )

            with state_lock:
                latest_result = processed

            print(
                json.dumps(
                    raw_result,
                    indent=2
                )
            )

            if processed["found"]:

                print(
                    f"Centroid: "
                    f"{processed['centroid']}"
                )

                print(
                    f"Area: "
                    f"{int(processed['area'])} px²"
                )

                print(
                    f"Error X: "
                    f"{processed['err_x']}"
                )

                print(
                    f"Error Y: "
                    f"{processed['err_y']}"
                )

            else:

                print(
                    f"{TARGET_OBJECT} not found."
                )

        except Exception as e:

            print()
            print("Gemini error:")
            print(e)

        finally:

            with state_lock:
                gemini_busy = False


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(frame, result):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Draw image center
    # --------------------------------------------------------

    image_cx = width // 2
    image_cy = height // 2

    cv2.drawMarker(
        frame,
        (image_cx, image_cy),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        20,
        1
    )

    # --------------------------------------------------------
    # No target
    # --------------------------------------------------------

    if result is None:

        cv2.putText(
            frame,
            f"Searching: {TARGET_OBJECT}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    # --------------------------------------------------------
    # Target not found
    # --------------------------------------------------------

    if not result["found"]:

        cv2.putText(
            frame,
            f"{TARGET_OBJECT}: NOT FOUND",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = result["box_pixels"]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 229, 160),
        2
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    cx, cy = result["centroid"]

    cv2.circle(
        frame,
        (cx, cy),
        6,
        (0, 255, 255),
        -1
    )

    # --------------------------------------------------------
    # Draw line from image center to target
    # --------------------------------------------------------

    cv2.line(
        frame,
        (image_cx, image_cy),
        (cx, cy),
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    cv2.putText(
        frame,
        result["class_name"],
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 229, 160),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Center: ({cx}, {cy})",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Pixel errors
    # --------------------------------------------------------

    err_x = result["err_x"]
    err_y = result["err_y"]

    cv2.putText(
        frame,
        f"Err X: {err_x}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Err Y: {err_y}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Area: {int(result['area'])}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Alignment status
    # --------------------------------------------------------

    centered_x = abs(err_x) <= DEAD_ZONE_X
    centered_y = abs(err_y) <= DEAD_ZONE_Y

    if centered_x and centered_y:

        status = "CENTERED"

    elif not centered_x:

        status = "MOVE X"

    else:

        status = "MOVE Y"

    cv2.putText(
        frame,
        status,
        (10, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    global latest_frame

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}"
        )

    # Match your robot camera resolution.

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
        30
    )

    print()
    print("==========================================")
    print(" Gemini Robotics-ER 2 + P-Control Test")
    print("==========================================")
    print()
    print(f"Target: {TARGET_OBJECT}")
    print(f"Gemini interval: {GEMINI_INTERVAL}s")
    print(f"Camera: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print()
    print("Gemini is NOT controlling the arm.")
    print("No serial commands are being sent.")
    print()
    print("Q = quit")
    print()

    # Start Gemini worker.

    worker = threading.Thread(
        target=gemini_worker,
        daemon=True
    )

    worker.start()

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera read failed.")
            break

        # Store newest frame for Gemini worker.

        with state_lock:
            latest_frame = frame.copy()
            result = latest_result

        # Draw latest Gemini result.

        display = draw_result(
            frame.copy(),
            result
        )

        # Top instruction.

        cv2.putText(
            display,
            "Gemini Robotics-ER 2",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Gemini Robotics Vision",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()