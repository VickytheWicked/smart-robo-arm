import cv2
import json
import os
import threading
import time

from google import genai
from google.genai import types


MODEL = "gemini-robotics-er-2-preview"

GEMINI_INTERVAL = 0.5


class GeminiRobotics:

    def __init__(self, target="eraser"):

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.target = target

        self.latest_frame = None
        self.latest_result = None

        self.lock = threading.Lock()

        self.running = False
        self.busy = False

        self.worker = None

        self.last_request = 0.0


    # ---------------------------------------------------------
    # Change target object
    # ---------------------------------------------------------

    def set_target(self, target):

        with self.lock:
            self.target = target

        print(
            f"[GEMINI] Target changed → {target}"
        )


    # ---------------------------------------------------------
    # Give Gemini the newest camera frame
    # ---------------------------------------------------------

    def update_frame(self, frame):

        with self.lock:
            self.latest_frame = frame.copy()


    # ---------------------------------------------------------
    # Get latest Gemini result
    # ---------------------------------------------------------

    def get_result(self):

        with self.lock:

            if self.latest_result is None:
                return None

            return self.latest_result.copy()


    # ---------------------------------------------------------
    # Gemini perception
    # ---------------------------------------------------------

    def _detect(self, frame, target):

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        if not success:
            raise RuntimeError(
                "Could not encode camera frame."
            )

        prompt = f"""
You are the vision system for a robotic arm.

The robot needs to pick:

"{target}"

Locate ONLY this target object.

Ignore every other object.

Return ONLY valid JSON:

{{
    "label": "{target}",
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
- If the target is not visible, return:

{{
    "label": "{target}",
    "box_2d": null
}}
"""

        response = self.client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=encoded.tobytes(),
                    mime_type="image/jpeg"
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        text = response.text.strip()

        # Remove markdown fences if present.
        if text.startswith("```"):

            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        result = json.loads(text)

        return result


    # ---------------------------------------------------------
    # Convert Gemini coordinates to robot result format
    # ---------------------------------------------------------

    def _process(self, result, frame):

        height, width = frame.shape[:2]

        box = result.get("box_2d")

        target = result.get(
            "label",
            self.target
        )

        # Target not found.

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

        if not isinstance(box, list) or len(box) != 4:

            raise RuntimeError(
                f"Invalid Gemini box: {box}"
            )

        # Gemini:
        #
        # [ymin, xmin, ymax, xmax]
        #

        ymin, xmin, ymax, xmax = box

        # 0-1000 → pixels

        x1 = int(
            float(xmin) / 1000 * width
        )

        y1 = int(
            float(ymin) / 1000 * height
        )

        x2 = int(
            float(xmax) / 1000 * width
        )

        y2 = int(
            float(ymax) / 1000 * height
        )

        # Clamp.

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

        # Centroid.

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Area.

        area = float(
            (x2 - x1) *
            (y2 - y1)
        )

        # Annotation.

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
            f"{target} [Gemini]",
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


    # ---------------------------------------------------------
    # Background worker
    # ---------------------------------------------------------

    def _worker(self):

        print("[GEMINI] Worker started.")

        while self.running:

            time.sleep(0.05)

            now = time.time()

            if (
                now - self.last_request
                < GEMINI_INTERVAL
            ):
                continue

            with self.lock:

                if self.busy:
                    continue

                if self.latest_frame is None:
                    continue

                frame = self.latest_frame.copy()
                target = self.target

                self.busy = True

            self.last_request = now

            try:

                print(
                    f"[GEMINI] Looking for: {target}"
                )

                raw = self._detect(
                    frame,
                    target
                )

                result = self._process(
                    raw,
                    frame
                )

                with self.lock:
                    self.latest_result = result

                if result["found"]:

                    print(
                        "[GEMINI] "
                        f"{result['class_name']} "
                        f"centroid="
                        f"{result['centroid']} "
                        f"area="
                        f"{int(result['area'])}"
                    )

                else:

                    print(
                        f"[GEMINI] "
                        f"{target} not found"
                    )

            except Exception as e:

                print(
                    f"[GEMINI] Error: {e}"
                )

            finally:

                with self.lock:
                    self.busy = False


    # ---------------------------------------------------------
    # Start worker
    # ---------------------------------------------------------

    def start(self):

        if self.running:
            return

        self.running = True

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.worker.start()


    # ---------------------------------------------------------
    # Stop worker
    # ---------------------------------------------------------

    def stop(self):

        self.running = False