"""
agent.py — Ollama Vision Agent for 6-DOF RoboArm (Raspberry Pi 5)

Run:
  python agent.py
  python agent.py --model llava:7b --interval 2.0 --verbose
  python agent.py --no-voice
"""

import cv2
import ollama
import base64
import serial
import threading
import time
import json
import argparse
import os
import sys
import requests
import pyttsx3
from datetime import datetime
import numpy as np
from vision_detection import detect_red_object, compute_pixel_error

# ── Config ────────────────────────────────────────────────────────────────────
SERIAL_PORT      = "/dev/ttyACM0"
BAUD_RATE        = 115200
CAMERA_INDEX     = 0
FRAME_W, FRAME_H = 640, 480
SEND_W,  SEND_H  = 512, 384
LOG_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_log.jsonl")
DEFAULT_MODEL    = "moondream"

OLLAMA_HOST      = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are the brain of a 6-DOF robotic arm sitting on a table. You see through "
    "a forward-facing camera mounted on the arm. You can rotate left and right, "
    "raise and lower your shoulder, and grab objects with your gripper.\n\n"
    "Your personality: curious, slightly dramatic, narrates everything like a nature "
    "documentary robot. You enjoy exploring and get excited when you spot red objects.\n\n"
    "Your only job each turn is to look at the image, decide what to do, and respond "
    "in this EXACT JSON format with no extra text, no markdown, no code fences:\n\n"
    "{\n"
    '  "thought": "your private reasoning about what you see",\n'
    '  "speech": "what you say out loud, max 2 short sentences, natural robot voice",\n'
    '  "command": "LEFT or RIGHT or FORWARD or BACK or CENTERED or NONE",\n'
    '  "target_visible": true or false,\n'
    '  "confidence": 0.0 to 1.0,\n'
    '  "note": "one short observation about the scene"\n'
    "}\n\n"
    "Command meanings:\n"
    "LEFT      = rotate base left to scan that direction\n"
    "RIGHT     = rotate base right to scan that direction\n"
    "FORWARD   = lower shoulder to move arm forward/down\n"
    "BACK      = raise shoulder to move arm back/up\n"
    "CENTERED  = ONLY send this when a red object is clearly visible AND\n"
    "            centered within roughly 80 pixels of the image center —\n"
    "            this triggers the full approach and grab sequence on the robot\n"
    "NONE      = hold position, keep observing\n\n"
    "Rules:\n"
    "- If you see a red object off-center, correct toward it with LEFT/RIGHT/FORWARD/BACK\n"
    "- If red object is centered, send CENTERED\n"
    "- If nothing interesting, explore slowly with LEFT or RIGHT\n"
    "- Never make up objects you cannot see\n"
    "- Keep speech under 20 words — it will be spoken aloud"
)


# ── Model resolution ──────────────────────────────────────────────────────────

def get_available_models() -> list:
    """Query Ollama for locally pulled models."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"[OLLAMA] Could not list models: {e}")
        return []


def resolve_model(requested: str) -> str:
    """
    Return the model to use:
    - If requested model is available, use it
    - Else pick the first available vision model from the list
    - Else exit with instructions
    """
    available = get_available_models()
    print(f"[OLLAMA] Available models: {available}")

    if any(requested in m for m in available):
        return requested

    vision_models = ["moondream", "llava", "bakllava", "minicpm-v"]
    for vm in vision_models:
        match = next((m for m in available if vm in m.lower()), None)
        if match:
            print(f"[OLLAMA] '{requested}' not found, using '{match}' instead")
            return match

    print("[OLLAMA] No vision model found. Run one of:")
    print("  ollama pull moondream")
    print("  ollama pull llava:7b")
    sys.exit(1)


# ── TTS engine (lazy-initialized on first use) ────────────────────────────────

_engine  = None
tts_lock = threading.Lock()
tts_busy = False


def _init_tts():
    global _engine
    eng = pyttsx3.init()
    eng.setProperty("rate", 155)
    voices = eng.getProperty("voices")
    for v in voices:
        if any(x in v.name.lower() for x in ["daniel", "david", "alex", "english"]):
            eng.setProperty("voice", v.id)
            break
    _engine = eng


def speak(text: str):
    global tts_busy

    def _run():
        global tts_busy
        with tts_lock:
            tts_busy = True
            _engine.say(text)
            _engine.runAndWait()
            tts_busy = False

    threading.Thread(target=_run, daemon=True).start()


# ── Serial ────────────────────────────────────────────────────────────────────

ser      = None
ser_lock = threading.Lock()


def connect_serial(port: str):
    global ser
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(1.5)
        print(f"[SERIAL] Connected: {port}")
    except Exception as e:
        print(f"[SERIAL] Failed: {e} — running in no-serial mode")
        ser = None


def send_cmd(cmd: str):
    if ser and ser.is_open:
        with ser_lock:
            ser.write((cmd + "\n").encode())
            ser.flush()
        print(f"[CMD] → {cmd}")
    else:
        print(f"[CMD] (no serial) → {cmd}")


# ── Frame capture and encoding ────────────────────────────────────────────────

def capture_frame(cap) -> tuple:
    ret, frame = cap.read()
    if not ret:
        return False, None, ""
    small = cv2.resize(frame, (SEND_W, SEND_H))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode("utf-8")
    return True, frame, b64


# ── Any-object CV detector ────────────────────────────────────────────────────

def detect_any_object(frame: np.ndarray) -> dict:
    """
    Find the largest visually distinct object in the frame using edge detection.
    Works for any colour — no HSV threshold needed.
    Returns: found, centroid (cx,cy), area, annotated frame.
    """
    h, w  = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 25, 80)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = dict(found=False, centroid=None, area=0.0, annotated=frame.copy())
    if not contours:
        return result

    # Keep contours that are a plausible object size (0.5 %–60 % of frame)
    frame_area = w * h
    valid = [c for c in contours
             if frame_area * 0.005 < cv2.contourArea(c) < frame_area * 0.60]
    if not valid:
        return result

    largest = max(valid, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    M       = cv2.moments(largest)
    if M["m00"] == 0:
        return result

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    ann = frame.copy()
    cv2.drawContours(ann, [largest], -1, (0, 200, 255), 2)
    cv2.circle(ann, (cx, cy), 7, (0, 200, 255), -1)
    cv2.line(ann, (w // 2, h // 2), (cx, cy), (255, 140, 0), 1)
    result.update(found=True, centroid=(cx, cy), area=area, annotated=ann)
    return result


def _cmd_from_centroid(cx: int, cy: int, frame_w: int, frame_h: int) -> tuple:
    """Map a centroid to (command, confidence)."""
    err_x = cx - frame_w // 2
    err_y = cy - frame_h // 2
    dead  = 80   # px dead-zone radius
    conf  = 0.72

    if abs(err_x) <= dead and abs(err_y) <= dead:
        return "CENTERED", 0.85
    if abs(err_x) >= abs(err_y):
        return ("RIGHT" if err_x > 0 else "LEFT"), conf
    return ("FORWARD" if err_y > 0 else "BACK"), conf


# ── Ollama call ───────────────────────────────────────────────────────────────

def _md_call(b64_image: str, question: str) -> str:
    """Single moondream call — returns stripped response text."""
    r = ollama.chat(
        model="moondream",
        messages=[{"role": "user", "content": question, "images": [b64_image]}],
    )
    return r["message"]["content"].strip()


def ask_ollama(b64_image: str, context: str, model: str, history: list,
               det: dict = None, frame: np.ndarray = None) -> tuple:
    """
    Returns (parsed_dict, raw_str).

    moondream strategy (two focused calls moondream can actually answer):
      Call 1 — "Describe what you see"       → speech narration
      Call 2 — "Is there a graspable object?" → True/False
      Position is computed from the raw frame via edge-based CV (any colour).

    llava/other — full JSON system prompt, parse response.
    """
    # ── moondream path ────────────────────────────────────────────────────
    if "moondream" in model.lower():
        desc      = "something on the surface"
        graspable = False
        raw       = ""
        try:
            desc = _md_call(
                b64_image,
                "Describe what you see in this image in one sentence."
            )
            raw = desc

            grab_ans  = _md_call(
                b64_image,
                "Is there an object a robot arm could pick up in this image? True or false."
            )
            raw      += " | " + grab_ans
            graspable = "true" in grab_ans.lower()

        except Exception as e:
            print(f"[OLLAMA ERROR] moondream call failed: {e}")

        # Position from edge-based CV (works for any colour)
        cmd, conf, target = "LEFT", 0.25, False
        if graspable and frame is not None:
            cv_det = detect_any_object(frame)
            if cv_det["found"]:
                cx, cy = cv_det["centroid"]
                cmd, conf = _cmd_from_centroid(cx, cy, FRAME_W, FRAME_H)
                target    = True

        # Safety gate on CENTERED
        if cmd == "CENTERED" and conf < 0.7:
            cmd  = "NONE"
            conf = 0.6

        speech = desc[:120] if desc else "Scanning."
        if not graspable:
            speech = desc[:80] + " — nothing to grab here." if desc else "Scanning."

        parsed = {
            "command":        cmd,
            "speech":         speech,
            "thought":        f"graspable={graspable} cv_cmd={cmd} conf={conf:.2f}",
            "target_visible": target,
            "confidence":     round(conf, 2),
            "note":           desc[:80],
        }
        return parsed, raw

    # ── llava / other models: full JSON prompt ─────────────────────────────
    user_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current sensor data: {context}\n"
        f"What do you see and what do you do? Respond ONLY in the JSON format above."
    )

    messages = []

    for past_user, past_assistant in history[-3:]:
        messages.append({"role": "user",      "content": past_user})
        messages.append({"role": "assistant", "content": past_assistant})

    messages.append({
        "role":    "user",
        "content": user_text,
        "images":  [b64_image],
    })

    raw = ""
    try:
        response = ollama.chat(model=model, messages=messages)
        raw = response["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in model response")

        parsed = json.loads(raw[start:end])

        parsed.setdefault("command",        "NONE")
        parsed.setdefault("speech",         "Observing.")
        parsed.setdefault("thought",        "")
        parsed.setdefault("target_visible", False)
        parsed.setdefault("confidence",     0.5)
        parsed.setdefault("note",           "")

        if parsed["command"] == "CENTERED" and parsed["confidence"] < 0.7:
            parsed["command"] = "NONE"
            parsed["speech"]  = "Almost. Let me look a little closer."

        return parsed, raw

    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        # If we have raw text (JSON parse failed), try natural-language extraction
        if raw:
            print("[OLLAMA] Falling back to natural-language parser")
            parsed = _parse_natural_language(raw, context)
            return parsed, raw
        fallback = {
            "command":        "LEFT",
            "speech":         "Let me look around.",
            "thought":        f"error: {e}",
            "target_visible": False,
            "confidence":     0.0,
            "note":           "connection or parse error",
        }
        return fallback, ""


# ── Logging ───────────────────────────────────────────────────────────────────

def log_turn(data: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


# ── Display overlay ───────────────────────────────────────────────────────────

def draw_overlay(frame, det_result, decision, model_name):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (180, 180, 180), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (180, 180, 180), 1)

    if det_result["found"]:
        x, y, bw, bh = det_result["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 80), 2)
        cv2.circle(frame, det_result["centroid"], 6, (0, 255, 80), -1)

    def h2bgr(hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))

    cmd_colors = {
        "LEFT":     "#4499ff",
        "RIGHT":    "#4499ff",
        "FORWARD":  "#ffaa00",
        "BACK":     "#ffaa00",
        "CENTERED": "#00ff88",
        "NONE":     "#888888",
    }

    cmd    = decision.get("command", "NONE")
    conf   = decision.get("confidence", 0.0)
    speech = decision.get("speech", "")
    note   = decision.get("note", "")

    cv2.putText(frame, f"CMD: {cmd}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                h2bgr(cmd_colors.get(cmd, "#ffffff")), 2)
    cv2.putText(frame, f"CONF: {conf:.2f}  MODEL: {model_name}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)
    cv2.rectangle(frame, (0, h - 52), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, f'"{speech}"',
                (8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    cv2.putText(frame, note,
                (8, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
    return frame


# ── Main agent loop ───────────────────────────────────────────────────────────

def run(args):
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
    except Exception:
        print("[OLLAMA] Server not reachable. Start it with: ollama serve")
        sys.exit(1)

    model = resolve_model(args.model)
    print(f"[AGENT] Using model: {model}")

    connect_serial(args.port)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    if not args.no_voice:
        _init_tts()
        speak("Arm agent online. Systems ready.")

    history           = []
    arm_state         = "IDLE"
    centered_cooldown = 0
    last_decision     = {}

    try:
        while True:
            tick_start = time.time()

            if centered_cooldown > 0:
                centered_cooldown -= 1
                ret, frame = cap.read()
                if ret and not args.headless:
                    remaining = centered_cooldown * args.interval
                    cv2.putText(frame, f"GRABBING — {remaining:.1f}s remaining",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 200, 255), 2)
                    cv2.imshow("RoboArm Agent", frame)
                if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(args.interval)
                continue

            ok, frame, b64 = capture_frame(cap)
            if not ok:
                time.sleep(0.5)
                continue

            det  = detect_red_object(frame)
            ex, ey = (
                compute_pixel_error(det["centroid"], frame.shape)
                if det["found"] else (0, 0)
            )

            context = (
                f"red_detected={det['found']}, "
                f"offset_from_center=({ex},{ey})px, "
                f"object_area={int(det['area'])}px2, "
                f"arm_state={arm_state}"
            )

            print(f"[AGENT] Thinking... (model={model})")
            decision, raw = ask_ollama(b64, context, model, history, det=det, frame=frame)

            if args.verbose:
                print(json.dumps(decision, indent=2))

            cmd    = decision.get("command", "NONE")
            speech = decision.get("speech", "")

            if speech and not args.no_voice:
                speak(speech)
            print(f"[SPEECH]  {speech}")
            print(f"[THOUGHT] {decision.get('thought', '')}")

            if cmd != "NONE":
                send_cmd(cmd)
            if cmd == "CENTERED":
                arm_state = "APPROACHING"
                centered_cooldown = max(1, int(4.0 / args.interval))
                if not args.no_voice:
                    speak("Target acquired. Initiating grab sequence.")

            history.append((f"[sensor:{context}]", raw or json.dumps(decision)))
            if len(history) > 3:
                history.pop(0)

            log_turn({
                "timestamp": datetime.now().isoformat(),
                "model":     model,
                "context":   context,
                "decision":  decision,
                "arm_state": arm_state,
            })

            if not args.headless:
                display = draw_overlay(
                    det["annotated"].copy(), det, decision, model
                )
                cv2.imshow("RoboArm Agent", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.time() - tick_start
            sleep_t = max(0.0, args.interval - elapsed)
            print(f"[TICK] {elapsed:.2f}s elapsed, sleeping {sleep_t:.2f}s\n")
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
        if not args.no_voice and _engine:
            _engine.stop()
        print("[AGENT] Shutdown complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ollama Vision Agent — RoboArm Pi 5")
    ap.add_argument("--port",     default=SERIAL_PORT,   help="Serial port to ESP32")
    ap.add_argument("--camera",   default=0,  type=int,  help="Camera device index")
    ap.add_argument("--model",    default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--interval", default=1.5, type=float, help="Seconds per tick")
    ap.add_argument("--no-voice", action="store_true",   help="Disable TTS")
    ap.add_argument("--headless", action="store_true",   help="Skip cv2.imshow (for SSH/no-display)")
    ap.add_argument("--verbose",  action="store_true",   help="Print full JSON each tick")
    run(ap.parse_args())
