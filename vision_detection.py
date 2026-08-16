"""
vision_detection.py
-------------------
Red object detection for the 6-DOF robotic arm (eraser, red block, etc.)
Used by visual_servoing.py on Raspberry Pi 5.

Standalone demo:
  python vision_detection.py
  python vision_detection.py --image test.jpg
  python vision_detection.py --tune    # interactive HSV tuner
"""

import cv2
import numpy as np
import argparse
import time

# ── HSV thresholds — tuned for red eraser, indoor lighting ──────────────
# Red wraps around hue=0/180, so two ranges are needed
RED_LOWER1 = np.array([0,   120,  70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120,  70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500      # px² — smaller blobs are ignored as noise
BLUR_KERNEL      = (11, 11)
MORPH_KERNEL     = np.ones((5, 5), np.uint8)

CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480


def build_red_mask(hsv: np.ndarray) -> np.ndarray:
    """Binary mask covering both red hue ranges, morphologically cleaned."""
    m1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    m2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,  MORPH_KERNEL)
    return mask


def detect_red_object(frame: np.ndarray) -> dict:
    """
    Detect the largest red object in a BGR frame.

    Returns:
        found    : bool
        centroid : (cx, cy) or None
        bbox     : (x, y, w, h) or None
        area     : float (px²)
        mask     : binary mask
        annotated: frame with drawings overlaid
    """
    result = dict(found=False, centroid=None, bbox=None,
                  area=0.0, mask=None, annotated=frame.copy())

    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = build_red_mask(hsv)
    result["mask"] = mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return result

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return result

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(largest)

    result.update(found=True, centroid=(cx, cy), bbox=(x, y, w, h), area=area)

    # ── Annotate ────────────────────────────────────────────────────────
    ann     = frame.copy()
    fh, fw  = frame.shape[:2]
    fcx, fcy = fw // 2, fh // 2

    # Bounding box
    cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 80), 2)
    # Contour
    cv2.drawContours(ann, [largest], -1, (0, 200, 255), 1)
    # Centroid
    cv2.circle(ann, (cx, cy), 7, (0, 255, 80), -1)
    cv2.circle(ann, (cx, cy), 9, (255, 255, 255), 1)
    # Line from frame center to object
    cv2.line(ann, (fcx, fcy), (cx, cy), (255, 140, 0), 1)
    # Labels
    err_x = cx - fcx
    err_y = cy - fcy
    cv2.putText(ann, f"({cx},{cy})",
                (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
    cv2.putText(ann, f"area={int(area)}px2",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 80), 1)
    cv2.putText(ann, f"ErrX:{err_x:+d} ErrY:{err_y:+d}",
                (8, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)

    result["annotated"] = ann
    return result

def detect_green_object(frame: np.ndarray) -> dict:
    """
    Detect the largest green object in a BGR frame.

    Returns:
        found    : bool
        centroid : (cx, cy) or None
        bbox     : (x, y, w, h) or None
        area     : float (px²)
        mask     : binary mask
        annotated: frame with drawings overlaid
    """

    result = dict(
        found=False,
        centroid=None,
        bbox=None,
        area=0.0,
        mask=None,
        annotated=frame.copy()
    )

    # Blur to reduce camera noise
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Green HSV range
    GREEN_LOWER = np.array([35, 80, 50])
    GREEN_UPPER = np.array([85, 255, 255])

    # Create mask
    mask = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

    # Clean mask
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        MORPH_KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        MORPH_KERNEL
    )

    result["mask"] = mask

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return result

    # Largest green region
    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    if area < MIN_CONTOUR_AREA:
        return result

    # Calculate centroid
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return result

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)

    result.update(
        found=True,
        centroid=(cx, cy),
        bbox=(x, y, w, h),
        area=area
    )

    # --------------------------------------------------
    # Annotation
    # --------------------------------------------------

    ann = frame.copy()

    fh, fw = frame.shape[:2]

    # Frame center
    fcx = fw // 2
    fcy = fh // 2

    # Bounding box
    cv2.rectangle(
        ann,
        (x, y),
        (x + w, y + h),
        (0, 255, 80),
        2
    )

    # Contour
    cv2.drawContours(
        ann,
        [largest],
        -1,
        (0, 200, 255),
        1
    )

    # Centroid
    cv2.circle(
        ann,
        (cx, cy),
        7,
        (0, 255, 80),
        -1
    )

    cv2.circle(
        ann,
        (cx, cy),
        9,
        (255, 255, 255),
        1
    )

    # Line from image center → object
    cv2.line(
        ann,
        (fcx, fcy),
        (cx, cy),
        (255, 140, 0),
        1
    )

    # Pixel error
    err_x = cx - fcx
    err_y = cy - fcy

    # Labels
    cv2.putText(
        ann,
        f"GREEN ({cx},{cy})",
        (x, max(y - 26, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 80),
        1
    )

    cv2.putText(
        ann,
        f"area={int(area)}px2",
        (x, max(y - 10, 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 200, 80),
        1
    )

    cv2.putText(
        ann,
        f"ErrX:{err_x:+d} ErrY:{err_y:+d}",
        (8, fh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 255, 100),
        2
    )

    result["annotated"] = ann

    return result


def detect_yellow_object(frame: np.ndarray) -> dict:
    """
    Detect the largest green object in a BGR frame.

    Returns:
        found    : bool
        centroid : (cx, cy) or None
        bbox     : (x, y, w, h) or None
        area     : float (px²)
        mask     : binary mask
        annotated: frame with drawings overlaid
    """

    result = dict(
        found=False,
        centroid=None,
        bbox=None,
        area=0.0,
        mask=None,
        annotated=frame.copy()
    )

    # Blur to reduce camera noise
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Green HSV range
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([30, 255, 255])

    # Create mask
    mask = cv2.inRange(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER
    )

    # Clean mask
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        MORPH_KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        MORPH_KERNEL
    )

    result["mask"] = mask

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return result

    # Largest green region
    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    if area < MIN_CONTOUR_AREA:
        return result

    # Calculate centroid
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return result

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)

    result.update(
        found=True,
        centroid=(cx, cy),
        bbox=(x, y, w, h),
        area=area
    )

    # --------------------------------------------------
    # Annotation
    # --------------------------------------------------

    ann = frame.copy()

    fh, fw = frame.shape[:2]

    # Frame center
    fcx = fw // 2
    fcy = fh // 2

    # Bounding box
    cv2.rectangle(
        ann,
        (x, y),
        (x + w, y + h),
        (0, 255, 80),
        2
    )

    # Contour
    cv2.drawContours(
        ann,
        [largest],
        -1,
        (0, 200, 255),
        1
    )

    # Centroid
    cv2.circle(
        ann,
        (cx, cy),
        7,
        (0, 255, 80),
        -1
    )

    cv2.circle(
        ann,
        (cx, cy),
        9,
        (255, 255, 255),
        1
    )

    # Line from image center → object
    cv2.line(
        ann,
        (fcx, fcy),
        (cx, cy),
        (255, 140, 0),
        1
    )

    # Pixel error
    err_x = cx - fcx
    err_y = cy - fcy

    # Labels
    cv2.putText(
        ann,
        f"GREEN ({cx},{cy})",
        (x, max(y - 26, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 80),
        1
    )

    cv2.putText(
        ann,
        f"area={int(area)}px2",
        (x, max(y - 10, 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 200, 80),
        1
    )

    cv2.putText(
        ann,
        f"ErrX:{err_x:+d} ErrY:{err_y:+d}",
        (8, fh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 255, 100),
        2
    )

    result["annotated"] = ann

    return result


def detect_white_object(frame: np.ndarray) -> dict:
    """
    Detect the largest green object in a BGR frame.

    Returns:
        found    : bool
        centroid : (cx, cy) or None
        bbox     : (x, y, w, h) or None
        area     : float (px²)
        mask     : binary mask
        annotated: frame with drawings overlaid
    """

    result = dict(
        found=False,
        centroid=None,
        bbox=None,
        area=0.0,
        mask=None,
        annotated=frame.copy()
    )

    # Blur to reduce camera noise
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Green HSV range
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 50, 255])

    # Create mask
    mask = cv2.inRange(
        hsv,
        WHITE_LOWER,
        WHITE_UPPER
    )

    # Clean mask
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        MORPH_KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        MORPH_KERNEL
    )

    result["mask"] = mask

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return result

    # Largest green region
    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    if area < MIN_CONTOUR_AREA:
        return result

    # Calculate centroid
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return result

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)

    result.update(
        found=True,
        centroid=(cx, cy),
        bbox=(x, y, w, h),
        area=area
    )

    # --------------------------------------------------
    # Annotation
    # --------------------------------------------------

    ann = frame.copy()

    fh, fw = frame.shape[:2]

    # Frame center
    fcx = fw // 2
    fcy = fh // 2

    # Bounding box
    cv2.rectangle(
        ann,
        (x, y),
        (x + w, y + h),
        (0, 255, 80),
        2
    )

    # Contour
    cv2.drawContours(
        ann,
        [largest],
        -1,
        (0, 200, 255),
        1
    )

    # Centroid
    cv2.circle(
        ann,
        (cx, cy),
        7,
        (0, 255, 80),
        -1
    )

    cv2.circle(
        ann,
        (cx, cy),
        9,
        (255, 255, 255),
        1
    )

    # Line from image center → object
    cv2.line(
        ann,
        (fcx, fcy),
        (cx, cy),
        (255, 140, 0),
        1
    )

    # Pixel error
    err_x = cx - fcx
    err_y = cy - fcy

    # Labels
    cv2.putText(
        ann,
        f"WHITE ({cx},{cy})",
        (x, max(y - 26, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 80),
        1
    )

    cv2.putText(
        ann,
        f"area={int(area)}px2",
        (x, max(y - 10, 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 200, 80),
        1
    )

    cv2.putText(
        ann,
        f"ErrX:{err_x:+d} ErrY:{err_y:+d}",
        (8, fh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 255, 100),
        2
    )

    result["annotated"] = ann

    return result
    
def compute_pixel_error(centroid: tuple, frame_shape: tuple):
    """
    Pixel offset from frame center.
    Positive err_x → object is RIGHT of center → send RIGHT to ESP32
    Positive err_y → object is BELOW center    → send FORWARD to ESP32
    """
    if centroid is None:
        return None, None
    fh, fw = frame_shape[:2]
    return centroid[0] - fw // 2, centroid[1] - fh // 2


def run_tuner(cap):
    """Interactive HSV trackbar tuner — press 'q' to quit and print values."""
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H Low",  "HSV Tuner",   0, 180, lambda x: None)
    cv2.createTrackbar("H High", "HSV Tuner",  10, 180, lambda x: None)
    cv2.createTrackbar("S Low",  "HSV Tuner", 120, 255, lambda x: None)
    cv2.createTrackbar("V Low",  "HSV Tuner",  70, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        hsv  = cv2.cvtColor(cv2.GaussianBlur(frame, BLUR_KERNEL, 0), cv2.COLOR_BGR2HSV)
        hl   = cv2.getTrackbarPos("H Low",  "HSV Tuner")
        hh   = cv2.getTrackbarPos("H High", "HSV Tuner")
        sl   = cv2.getTrackbarPos("S Low",  "HSV Tuner")
        vl   = cv2.getTrackbarPos("V Low",  "HSV Tuner")
        mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, 255, 255]))
        cv2.imshow("HSV Tuner", mask)
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nTuned values:")
            print(f"RED_LOWER1 = ({hl}, {sl}, {vl})")
            print(f"RED_UPPER1 = ({hh}, 255, 255)")
            break
    cv2.destroyWindow("HSV Tuner")


def run_live(camera_idx: int = CAMERA_INDEX):
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return

    print("[INFO] Red Object Detector — 'q' quit  't' tune  'r' reset")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        result  = detect_red_object(frame)
        display = result["annotated"]

        # FPS + status
        cv2.putText(display, f"FPS:{fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        status_col = (0, 220, 60) if result["found"] else (40, 40, 200)
        cv2.putText(display, "FOUND" if result["found"] else "SEARCHING",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        # Mask inset (top-right)
        if result["mask"] is not None:
            ms = cv2.resize(result["mask"], (160, 120))
            display[8:128, FRAME_W - 168:FRAME_W - 8] = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Red Detector", display)

        if result["found"]:
            ex, ey = compute_pixel_error(result["centroid"], frame.shape)
            print(f"\r[DETECT] centroid={result['centroid']}  "
                  f"err=({ex:+d},{ey:+d})  area={int(result['area'])}px²", end="")

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('t'):
            run_tuner(cap)

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  default=None,        help="Static image path")
    ap.add_argument("--camera", default=0, type=int, help="Camera index")
    ap.add_argument("--tune",   action="store_true", help="Open HSV tuner")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot load {args.image}")
        else:
            r = detect_red_object(frame)
            print(f"found={r['found']}  centroid={r['centroid']}  "
                  f"bbox={r['bbox']}  area={r['area']:.0f}")
            cv2.imshow("Result", r["annotated"])
            cv2.imshow("Mask",   r["mask"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_live(args.camera)
