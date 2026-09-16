import asyncio
import cv2
import json
import os
import re
import threading
import time

from google import genai
from google.genai import types


MODEL = "gemini-robotics-er-2-streaming-preview"

# Don't send every 30-FPS frame to Gemini.
# The Live session stays open, but perception input is throttled.
FRAME_INTERVAL = 0.15


class GeminiRobotics:

    def __init__(self, target="eraser"):

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(api_key=api_key)

        self.target = target

        # Latest camera frame
        self.latest_frame = None

        # Latest Gemini result
        self.latest_result = None

        # Incremented ONLY when Gemini produces a new result
        self.result_version = 0

        self.lock = threading.Lock()

        self.running = False

        self.worker = None

        self.session = None

        self.frame_event = threading.Event()

        self.last_sent_frame = 0.0

    # =========================================================
    # TARGET
    # =========================================================

    def set_target(self, target):

        with self.lock:
            self.target = target

        print(f"[GEMINI] Target changed → {target}")

    # =========================================================
    # CAMERA INPUT
    # =========================================================

    def update_frame(self, frame):

        with self.lock:
            self.latest_frame = frame.copy()

        self.frame_event.set()

    # =========================================================
    # RESULT
    # =========================================================

    def get_result(self):

        with self.lock:

            if self.latest_result is None:
                return None

            return self.latest_result.copy()

    # =========================================================
    # RESULT VERSION
    # =========================================================

    def get_result_version(self):

        with self.lock:
            return self.result_version

    # =========================================================
    # PROMPT
    # =========================================================

    def _prompt(self):

        with self.lock:
            target = self.target

        return f"""
You are the vision system for a robotic arm.

The robot is trying to pick this object:

{target}

For each image, locate ONLY this object.

Return ONLY JSON.

Format:

{{
    "label": "{target}",
    "box_2d": [ymin, xmin, ymax, xmax]
}}

Coordinates are normalized from 0 to 1000.

box_2d means:

[ymin, xmin, ymax, xmax]

Rules:

- Detect only the requested object.
- Do not include the table.
- Do not include nearby objects.
- Do not merge objects.
- Do not hallucinate an object.
- If the object is not visible, return:

{{
    "label": "{target}",
    "box_2d": null
}}

Do not explain.
Do not use markdown.
Return JSON only.
"""

    # =========================================================
    # PARSE JSON
    # =========================================================

    def _extract_json(self, text):

        text = text.strip()

        # Remove markdown fences
        text = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = text.replace("```", "").strip()

        # Try direct JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # =========================================================
    # PROCESS BOX
    # =========================================================

    def _process(self, raw, frame):

        height, width = frame.shape[:2]

        with self.lock:
            target = self.target

        box = raw.get("box_2d")

        # -----------------------------------------------------
        # NOT FOUND
        # -----------------------------------------------------

        if box is None:

            return {
                "found": False,
                "centroid": None,
                "area": 0.0,
                "annotated": frame.copy(),
                "class_id": None,
                "class_name": target,
                "confidence": 0.0
            }

        # -----------------------------------------------------
        # VALIDATE
        # -----------------------------------------------------

        if not isinstance(box, list) or len(box) != 4:

            raise RuntimeError(
                f"Invalid Gemini box: {box}"
            )

        ymin, xmin, ymax, xmax = box

        # -----------------------------------------------------
        # 0-1000 → PIXELS
        # -----------------------------------------------------

        x1 = int(float(xmin) / 1000.0 * width)
        y1 = int(float(ymin) / 1000.0 * height)

        x2 = int(float(xmax) / 1000.0 * width)
        y2 = int(float(ymax) / 1000.0 * height)

        # -----------------------------------------------------
        # CLAMP
        # -----------------------------------------------------

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))

        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:

            return {
                "found": False,
                "centroid": None,
                "area": 0.0,
                "annotated": frame.copy(),
                "class_id": None,
                "class_name": target,
                "confidence": 0.0
            }

        # -----------------------------------------------------
        # CENTROID
        # -----------------------------------------------------

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        area = float(
            (x2 - x1) *
            (y2 - y1)
        )

        # -----------------------------------------------------
        # ANNOTATION
        # -----------------------------------------------------

        annotated = frame.copy()

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 229, 160),
            2
        )

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
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 229, 160),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"({cx},{cy})",
            (x1, min(height - 10, y2 + 20)),
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

    # =========================================================
    # PUBLISH NEW RESULT
    # =========================================================

    def _publish_result(self, result):

        with self.lock:

            self.latest_result = result

            # IMPORTANT:
            # increment ONLY when Gemini actually
            # produced a new perception result.
            self.result_version += 1

    # =========================================================
    # SEND FRAMES
    # =========================================================

    async def _sender(self, session):

        while self.running:

            self.frame_event.wait(
                timeout=0.05
            )

            self.frame_event.clear()

            now = time.time()

            if (
                now - self.last_sent_frame
                < FRAME_INTERVAL
            ):
                continue

            with self.lock:

                if self.latest_frame is None:
                    continue

                frame = self.latest_frame.copy()

            self.last_sent_frame = now

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    70
                ]
            )

            if not success:
                continue

            try:

                await session.send_realtime_input(
                    media=types.Blob(
                        data=encoded.tobytes(),
                        mime_type="image/jpeg"
                    )
                )

            except Exception as e:

                print(
                    f"[GEMINI] Frame send error: {e}"
                )

                break

    # =========================================================
    # RECEIVE GEMINI
    # =========================================================

    async def _receiver(self, session):

        try:

            async for response in session.receive():

                if not self.running:
                    break

                if not response.text:
                    continue

                text = response.text.strip()

                raw = self._extract_json(text)

                if raw is None:

                    print(
                        f"[GEMINI] Non-JSON response: {text}"
                    )

                    continue

                with self.lock:

                    if self.latest_frame is None:
                        continue

                    # Use the latest frame available when
                    # the response arrives.
                    frame = self.latest_frame.copy()

                try:

                    result = self._process(
                        raw,
                        frame
                    )

                    self._publish_result(
                        result
                    )

                    if result["found"]:

                        print(
                            "[GEMINI]",
                            result["class_name"],
                            "centroid=",
                            result["centroid"],
                            "area=",
                            int(result["area"]),
                            "version=",
                            self.result_version
                        )

                    else:

                        print(
                            "[GEMINI]",
                            self.target,
                            "not found",
                            "version=",
                            self.result_version
                        )

                except Exception as e:

                    print(
                        f"[GEMINI] Processing error: {e}"
                    )

        except Exception as e:

            print(
                f"[GEMINI] Receiver error: {e}"
            )

    # =========================================================
    # STREAMING SESSION
    # =========================================================

    async def _run_stream(self):

        print(
            "[GEMINI] Connecting to ER-2 Streaming..."
        )

        try:

            async with self.client.aio.live.connect(
                model=MODEL,
                config=types.LiveConnectConfig(
                    response_modalities=["TEXT"]
                )
            ) as session:

                self.session = session

                print(
                    "[GEMINI] ER-2 Streaming connected."
                )

                # Initial instruction

                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=self._prompt()
                                )
                            ]
                        )
                    ],
                    turn_complete=True
                )

                sender = asyncio.create_task(
                    self._sender(session)
                )

                receiver = asyncio.create_task(
                    self._receiver(session)
                )

                await asyncio.gather(
                    sender,
                    receiver
                )

        except Exception as e:

            print(
                f"[GEMINI] Streaming connection error: {e}"
            )

        finally:

            self.session = None

    # =========================================================
    # THREAD ENTRY
    # =========================================================

    def _worker(self):

        try:

            asyncio.run(
                self._run_stream()
            )

        except Exception as e:

            print(
                f"[GEMINI] Worker error: {e}"
            )

    # =========================================================
    # START
    # =========================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.worker.start()

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        self.running = False

        self.frame_event.set()

        print(
            "[GEMINI] Streaming stopped."
        )