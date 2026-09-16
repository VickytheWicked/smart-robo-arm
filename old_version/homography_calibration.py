#!/usr/bin/env python3
"""
Robotic Arm - Homography Calibration

Purpose:
    Calibrate the mapping from camera pixel coordinates (u, v)
    to physical tabletop coordinates (X, Y) in millimeters.

Physical marker coordinates:
    BLUE   = (0, 195)
    WHITE  = (0, 0)
    ORANGE = (370, 195)
    BLACK  = (370, 0)

Camera:
    Index = 1
    Resolution = 640x480

Controls:
    Click the CENTER of each marker in this order:
        1. BLUE
        2. WHITE
        3. ORANGE
        4. BLACK

    After calibration:
        Left click anywhere in the workspace -> converts pixel to table mm
        R -> reset the four calibration clicks
        S -> save calibration
        Q / ESC -> quit

The camera must remain fixed after calibration.
"""

import cv2
import numpy as np
import json
from pathlib import Path

CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WINDOW_NAME = "Robotic Arm - Homography Calibration"

# IMPORTANT:
# These are the physical coordinates in millimeters.
# Ordering must match IMAGE_POINTS_ORDER below.
TABLE_POINTS = np.array(
    [
        [0.0, 195.0],    # BLUE
        [0.0, 0.0],      # WHITE
        [370.0, 195.0],  # ORANGE
        [370.0, 0.0],    # BLACK
    ],
    dtype=np.float32,
)

POINT_NAMES = ["BLUE", "WHITE", "ORANGE", "BLACK"]

SAVE_JSON = Path("homography_calibration.json")
SAVE_NPY = Path("homography_matrix.npy")

clicked_points = []
H = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, H

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # If calibration is complete, clicking performs a test conversion.
    if H is not None:
        pixel = np.array([[[float(x), float(y)]]], dtype=np.float32)
        table = cv2.perspectiveTransform(pixel, H)[0, 0]

        print(
            f"[TEST] pixel=({x}, {y}) -> "
            f"table=({table[0]:.2f}, {table[1]:.2f}) mm"
        )
        return

    if len(clicked_points) >= 4:
        return

    clicked_points.append((x, y))
    name = POINT_NAMES[len(clicked_points) - 1]

    print(f"[CLICK] {name}: ({x}, {y})")

    if len(clicked_points) == 4:
        image_points = np.array(clicked_points, dtype=np.float32)

        H, status = cv2.findHomography(
            image_points,
            TABLE_POINTS,
            method=0
        )

        if H is None:
            print("[ERROR] Homography calculation failed.")
            clicked_points.clear()
            return

        print("\n========== HOMOGRAPHY ==========")
        print(H)
        print("================================\n")

        print("[INFO] Calibration complete.")
        print("[INFO] Left-click anywhere to test pixel -> table conversion.")
        print("[INFO] Press S to save calibration.")
        print("[INFO] Press R to recalibrate.")


def draw_ui(frame):
    output = frame.copy()

    # Draw selected calibration points.
    for i, (x, y) in enumerate(clicked_points):
        cv2.circle(output, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            output,
            POINT_NAMES[i],
            (x + 10, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # NO TOP HEADER.
    # A small status strip at the very bottom leaves the camera workspace clear.
    panel_h = 42
    cv2.rectangle(
        output,
        (0, FRAME_HEIGHT - panel_h),
        (FRAME_WIDTH, FRAME_HEIGHT),
        (0, 0, 0),
        -1,
    )

    if H is None:
        if len(clicked_points) < 4:
            status = f"CLICK {POINT_NAMES[len(clicked_points)]} CENTER ({len(clicked_points)}/4)"
        else:
            status = "CALCULATING..."
    else:
        status = "CALIBRATED - CLICK ANYWHERE TO TEST"

    cv2.putText(
        output,
        status,
        (8, FRAME_HEIGHT - 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "R reset | S save | Q/ESC quit",
        (8, FRAME_HEIGHT - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return output


def save_calibration():
    if H is None:
        print("[ERROR] No homography to save.")
        return

    data = {
        "camera_index": CAMERA_INDEX,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "point_order": POINT_NAMES,
        "image_points": [
            [int(x), int(y)] for x, y in clicked_points
        ],
        "table_points_mm": TABLE_POINTS.tolist(),
        "homography": H.tolist(),
    }

    with SAVE_JSON.open("w") as f:
        json.dump(data, f, indent=2)

    np.save(SAVE_NPY, H)

    print(f"[SAVED] {SAVE_JSON.resolve()}")
    print(f"[SAVED] {SAVE_NPY.resolve()}")


def main():
    global clicked_points, H

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Check the camera index and whether another program is using it."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("==============================================")
    print(" ROBOTIC ARM HOMOGRAPHY CALIBRATION")
    print("==============================================")
    print(f"Camera index : {CAMERA_INDEX}")
    print(f"Resolution   : {actual_w} x {actual_h}")
    print()
    print("CLICK ORDER:")
    for i, name in enumerate(POINT_NAMES, 1):
        xy = TABLE_POINTS[i - 1]
        print(f"  {i}. {name:6s} -> ({xy[0]:.0f}, {xy[1]:.0f}) mm")
    print()
    print("Click the CENTER of each marker.")
    print("Do NOT move the camera or markers.")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read camera frame.")
            break

        # Force the working frame size if the camera returned another size.
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        display = draw_ui(frame)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord("r"):
            clicked_points.clear()
            H = None
            print("\n[RESET] Calibration points cleared.\n")

        elif key == ord("s"):
            save_calibration()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
