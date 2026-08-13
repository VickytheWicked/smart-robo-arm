"""
visual_servoing.py
------------------
Raspberry Pi 5 — Visual Servoing Controller
Reads camera, detects red object, sends serial commands to XIAO ESP32-C3.

Matches the state machine in visual_servo/main.cpp exactly:
  ALIGN    → send LEFT / RIGHT / FORWARD / BACK
  CENTERED → send CENTERED (ESP32 takes over approach + grab)

Serial: /dev/ttyACM0, 115200 baud, newline-terminated ASCII

Usage:
  python visual_servoing.py
  python visual_servoing.py --port /dev/ttyUSB0 --camera 1
"""

import os
import cv2
import serial
import threading
import time
import argparse
from vision_detection import detect_red_object, compute_pixel_error

# ── Config ────────────────────────────────────────────────
SERIAL_PORT   = "/dev/ttyACM0"
BAUD_RATE     = 115200
CAMERA_INDEX  = 0
FRAME_W       = 640
FRAME_H       = 480

# Dead-zone: within this many pixels from center → object is "centered"
DEAD_ZONE_X   = 30   # px
DEAD_ZONE_Y   = 30   # px

# How long to wait between sending commands (throttle)
CMD_INTERVAL  = 0.08  # seconds (~12 Hz max)

# After sending CENTERED, wait this long before re-evaluating
# (ESP32 handles approach/grab autonomously during this window)
CENTERED_HOLD = 0.5   # seconds

# ── Globals ───────────────────────────────────────────────
serial_lock   = threading.Lock()
esp_log_lines = []   # lines received from ESP32
running       = True


def serial_reader(ser):
    """Background thread: reads ESP32 debug prints (Distance:, State:, etc.)"""
    global esp_log_lines
    while running:
        try:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    esp_log_lines.append(line)
                    if len(esp_log_lines) > 50:
                        esp_log_lines.pop(0)
                    print(f"[ESP32] {line}")
        except Exception:
            pass
        time.sleep(0.005)


def send_cmd(ser, cmd: str):
    """Send a newline-terminated command to ESP32."""
    with serial_lock:
        try:
            ser.write((cmd + "\n").encode("utf-8"))
            ser.flush()
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] {e}")


def decide_command(err_x, err_y) -> str | None:
    """
    Map pixel error to serial command.
    Matches firmware movement functions exactly:
      LEFT    → J0 angle +2 (base turns left)
      RIGHT   → J0 angle -2 (base turns right)
      FORWARD → J1 angle -2 (shoulder forward/down)
      BACK    → J1 angle +2 (shoulder back/up)

    Priority: horizontal alignment first (base), then vertical (shoulder).
    Returns None if within dead zone on both axes.
    """
    in_x = abs(err_x) <= DEAD_ZONE_X
    in_y = abs(err_y) <= DEAD_ZONE_Y

    if in_x and in_y:
        return "CENTERED"

    # Horizontal dominates — align base first
    if not in_x:
        # Camera is mounted so that object right of center → arm needs to turn right
        if err_x > 0:
            return "RIGHT"
        else:
            return "LEFT"

    # Vertical — align shoulder
    if not in_y:
        if err_y > 0:
            return "FORWARD"   # object below center → lower shoulder
        else:
            return "BACK"      # object above center → raise shoulder

    return None


def draw_hud(frame, result, cmd, err_x, err_y, mode_state, fps):
    """Draw status overlay on the frame for monitoring."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Cross-hair
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

    # Dead zone rectangle
    cv2.rectangle(frame,
                  (cx - DEAD_ZONE_X, cy - DEAD_ZONE_Y),
                  (cx + DEAD_ZONE_X, cy + DEAD_ZONE_Y),
                  (80, 80, 80), 1)

    # Bounding box + centroid from detection
    if result["found"]:
        x, y, bw, bh = result["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 80), 2)
        px, py = result["centroid"]
        cv2.circle(frame, (px, py), 6, (0, 255, 80), -1)
        cv2.line(frame, (cx, cy), (px, py), (255, 140, 0), 1)

    # HUD text
    color_found = (0, 255, 80) if result["found"] else (60, 60, 220)
    cv2.putText(frame, f"DETECT: {'YES' if result['found'] else 'NO'}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_found, 2)
    cv2.putText(frame, f"CMD: {cmd or 'NONE'}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    cv2.putText(frame, f"ErrX: {err_x:+d}  ErrY: {err_y:+d}",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"STATE: {mode_state}",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run(port: str, camera_idx: int):
    global running
    show_window = _has_display()

    # ── Open serial ─────────────────────────────────────
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)   # Let ESP32 boot / reset settle
        print(f"[SERIAL] Connected to {port} @ {BAUD_RATE}")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open serial port {port}: {e}")
        print("  Check: ls /dev/tty* | grep -E 'ACM|USB'")
        return

    # Background serial reader thread
    t = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    t.start()

    # ── Open camera ──────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        ser.close()
        return

    print("[INFO] Visual servoing running. Press 'q' to quit (window mode) or Ctrl+C (headless).")
    print(f"[INFO] Dead zone: ±{DEAD_ZONE_X}px X, ±{DEAD_ZONE_Y}px Y")
    if not show_window:
        print("[INFO] No display detected — running headless (no preview window).")

    last_cmd_time  = 0
    last_cmd       = None
    pi_state       = "ALIGN"     # mirrors ESP32 state machine
    prev_time      = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            result = detect_red_object(frame)
            err_x, err_y = 0, 0

            if result["found"]:
                err_x, err_y = compute_pixel_error(result["centroid"], frame.shape)

            # Determine command to send
            cmd = None
            if result["found"] and (now - last_cmd_time) >= CMD_INTERVAL:
                if pi_state == "ALIGN":
                    cmd = decide_command(err_x, err_y)

                    if cmd == "CENTERED":
                        pi_state = "CENTERED_SENT"
                        send_cmd(ser, "CENTERED")
                        print("[PI] Object centered — sending CENTERED")
                        last_cmd_time = now
                        last_cmd = "CENTERED"
                        time.sleep(CENTERED_HOLD)
                        # After this, ESP32 runs APPROACH → GRAB → DONE autonomously
                        # Pi monitors serial output; reset when done
                        pi_state = "WAITING_DONE"

                    elif cmd and cmd != last_cmd:
                        send_cmd(ser, cmd)
                        print(f"[PI→ESP32] {cmd}  (err_x={err_x:+d}, err_y={err_y:+d})")
                        last_cmd_time = now
                        last_cmd = cmd

                elif pi_state == "WAITING_DONE":
                    # Check if ESP32 printed "DONE" or distance ≤ 6 → grab complete
                    for line in esp_log_lines[-10:]:
                        if "DONE" in line or "GRAB" in line:
                            print("[PI] Grab complete — resetting to ALIGN")
                            pi_state = "ALIGN"
                            last_cmd = None
                            break
                    # Timeout fallback: if object disappears, reset
                    if not result["found"]:
                        pi_state = "ALIGN"
                        last_cmd = None

            elif not result["found"]:
                # Object lost — reset Pi state, ESP32 resets on non-CENTERED cmd
                if pi_state != "ALIGN":
                    pi_state = "ALIGN"
                    last_cmd = None

            if show_window:
                display = draw_hud(result["annotated"].copy(),
                                   result, cmd or last_cmd,
                                   err_x, err_y, pi_state, fps)
                cv2.imshow("Visual Servoing", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    pi_state = "ALIGN"
                    last_cmd = None
                    print("[PI] Manual reset → ALIGN")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        print("[INFO] Serial closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Servoing — Pi side")
    parser.add_argument("--port",   default=SERIAL_PORT,  help="Serial port")
    parser.add_argument("--camera", default=CAMERA_INDEX, type=int, help="Camera index")
    args = parser.parse_args()
    run(args.port, args.camera)
