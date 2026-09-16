import requests
import urllib3

# Suppress warnings caused by the local self-signed certificate
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


class RobotArm:
    """
    HTTP client for the Robot Arm dashboard server (server_v9.py).
    Supports manual joint control, directional motion, gripper control,
    and dynamic target object visual servoing (pick_object).
    """

    def __init__(self, server_url="https://127.0.0.1:3000"):
        self.server_url = server_url.rstrip("/")

    # =====================================================
    # DYNAMIC TARGET PICKING & COLOR VISION OFFLOADING
    # =====================================================

    def pick_color_object(self, color: str, item: str = ""):
        """
        Commands the robot arm to locate, track, and pick up
        a colored object (e.g. red lego brick, green block, yellow brick)
        by offloading to the high-speed local vision_detection system.
        """
        color = str(color).strip().lower()
        item = str(item or "object").strip()
        display_name = f"{color} {item}".strip()

        print(f"🎯 ROBOT COLOR PICK REQUEST: '{color}' ({display_name}) via vision_detection")

        try:
            response = requests.post(
                f"{self.server_url}/api/pick",
                json={"target": display_name, "color": color},
                timeout=10,
                verify=False
            )

            if response.ok:
                data = response.json()
                if data.get("ok"):
                    return f"Offloading task to vision detection for the {display_name}. Local 30 FPS color visual servoing engaged."
                return f"Robot server rejected pick request for '{display_name}': {data.get('error', 'unknown error')}"

            return f"Robot server returned status code {response.status_code} for color pick request."

        except requests.RequestException as e:
            print(f"❌ Robot color pick API error: {e}")
            return f"Could not communicate with the robot server: {e}"

    def pick_object(self, target: str, color: str = ""):
        """
        Commands the robot arm to locate, track, and pick up
        any specified target object. Automatically offloads to
        vision_detection if a supported color is present.
        """
        target = str(target).strip()
        color = str(color or "").strip().lower()
        if not target and not color:
            return "Please provide an object name or color to pick."

        payload = {"target": target}
        if color:
            payload["color"] = color

        print(f"🎯 ROBOT PICK REQUEST: '{target}' (color={color or 'auto'})")

        try:
            response = requests.post(
                f"{self.server_url}/api/pick",
                json=payload,
                timeout=10,
                verify=False
            )

            if response.ok:
                data = response.json()
                if data.get("ok"):
                    src = data.get("source", "gemini")
                    if src == "vision_detection":
                        detected_col = data.get("color", color)
                        return f"Offloading task to local vision detection for the {target or detected_col}. Color visual servoing engaged."
                    return f"Visual servoing activated to pick the '{target}'. Tracking engaged."
                return f"Robot server rejected pick request for '{target}': {data.get('error', 'unknown error')}"

            return f"Robot server returned status code {response.status_code} for pick request."

        except requests.RequestException as e:
            print(f"❌ Robot pick API error: {e}")
            return f"Could not communicate with the robot server: {e}"

    def stop_pick(self):
        """
        Stops active visual servoing / picking operation.
        """
        print("🛑 STOPPING VISUAL SERVO PICK")

        try:
            response = requests.post(
                f"{self.server_url}/api/stop-pick",
                timeout=10,
                verify=False
            )

            if response.ok:
                return "Visual servoing and object picking has been stopped."

            return f"Robot server returned status code {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Robot stop pick error: {e}")
            return f"Could not communicate with the robot server: {e}"

    def stop_tracking(self):
        """
        Stops active object visual tracking, visual servoing, and halts frame processing.
        """
        print("🛑 STOPPING OBJECT TRACKING")

        try:
            response = requests.post(
                f"{self.server_url}/api/stop-tracking",
                timeout=10,
                verify=False
            )

            if response.ok:
                return "Object tracking has been stopped. Camera frame processing halted."

            return f"Robot server returned status code {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Robot stop tracking error: {e}")
            return f"Could not communicate with the robot server: {e}"

    # =====================================================
    # DIRECTIONAL COMMANDS
    # =====================================================

    def move_arm(self, command):
        command = command.upper().strip()

        valid_commands = {
            "LEFT",
            "RIGHT",
            "FORWARD",
            "BACK",
            "CENTERED"
        }

        if command not in valid_commands:
            return f"Invalid robot command: {command}"

        print(f"🤖 ROBOT COMMAND: {command}")

        try:
            response = requests.post(
                f"{self.server_url}/api/command",
                json={"cmd": command},
                timeout=3,
                verify=False
            )

            if response.ok:
                data = response.json()
                if data.get("ok"):
                    return f"Robot arm command {command} executed successfully."
                return f"Robot rejected command {command}."

            return f"Robot server returned {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Robot server error: {e}")
            return f"Could not communicate with the robot server: {e}"

    # =====================================================
    # RESET ARM
    # =====================================================

    def reset_arm(self):
        print("🔄 RESETTING ROBOT ARM")

        try:
            response = requests.post(
                f"{self.server_url}/api/reset",
                timeout=3,
                verify=False
            )

            if response.ok:
                print("✅ Robot reset")
                return "The robotic arm has been reset."

            print(f"❌ Reset failed: {response.status_code}")
            return f"Failed to reset the robotic arm. Server returned {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Robot server error: {e}")
            return f"Failed to reset the robotic arm: {e}"

    # =====================================================
    # GRIPPER CONTROL
    # =====================================================

    def open_gripper(self):
        print("🖐️ OPENING GRIPPER")

        try:
            response = requests.post(
                f"{self.server_url}/api/gripper/open",
                timeout=3,
                verify=False
            )

            if response.ok:
                data = response.json()
                if data.get("ok"):
                    return "The gripper has been opened."
                return "The gripper could not be opened."

            return f"Gripper open command failed. HTTP {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Gripper error: {e}")
            return f"Failed to open the gripper: {e}"

    def close_gripper(self):
        print("✊ CLOSING GRIPPER")

        try:
            response = requests.post(
                f"{self.server_url}/api/gripper/close",
                timeout=3,
                verify=False
            )

            if response.ok:
                data = response.json()
                if data.get("ok"):
                    return "The gripper has been closed."
                return "The gripper could not be closed."

            return f"Gripper close command failed. HTTP {response.status_code}."

        except requests.RequestException as e:
            print(f"❌ Gripper error: {e}")
            return f"Failed to close the gripper: {e}"

    # =====================================================
    # INDIVIDUAL JOINT CONTROL
    # =====================================================

    def move_joint(self, servo, angle):
        servo = int(servo)
        angle = int(angle)

        if not 0 <= servo <= 5:
            return "Invalid joint. The arm has joints 0 through 5."

        if not 0 <= angle <= 180:
            return "Invalid angle. The angle must be between 0 and 180 degrees."

        print(f"🤖 Moving joint {servo} to {angle} degrees")

        try:
            response = requests.post(
                f"{self.server_url}/api/servo",
                json={"servo": servo, "angle": angle},
                timeout=3,
                verify=False
            )

            if not response.ok:
                return f"Servo command failed. HTTP {response.status_code}."

            data = response.json()
            if data.get("ok"):
                return f"Joint {servo} moved to {angle} degrees."

            return f"Joint {servo} could not be moved."

        except requests.RequestException as e:
            print(f"❌ Servo API error: {e}")
            return f"Could not communicate with the robot: {e}"
