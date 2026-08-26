import requests
import urllib3

# Suppress warnings caused by the local self-signed certificate
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


class RobotArm:

    def __init__(self, server_url="https://127.0.0.1:3000"):
        self.server_url = server_url.rstrip("/")

    # =====================================================
    # STOP ARM
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
            return (
                f"Invalid robot command: {command}"
            )

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
                    return (
                        f"Robot arm command "
                        f"{command} executed successfully."
                    )

                return (
                    f"Robot rejected command "
                    f"{command}."
                )

            return (
                f"Robot server returned "
                f"{response.status_code}."
            )

        except requests.RequestException as e:

            print(f"❌ Robot server error: {e}")

            return (
                f"Could not communicate with "
                f"the robot server: {e}"
            )

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

                return (
                    "The robotic arm has been reset."
                )

            print(
                f"❌ Reset failed: "
                f"{response.status_code}"
            )

            return (
                f"Failed to reset the robotic arm. "
                f"Server returned "
                f"{response.status_code}."
            )

        except requests.RequestException as e:

            print(
                f"❌ Robot server error: {e}"
            )

            return (
                f"Failed to reset the robotic arm: {e}"
            )

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

                    return (
                        "The robotic arm has been reset."
                    )

                print(
                    f"❌ Reset failed: "
                    f"{response.status_code}"
                )

                return (
                    f"Failed to reset the robotic arm. "
                    f"Server returned "
                    f"{response.status_code}."
                )

            except requests.RequestException as e:

                print(
                    f"❌ Robot server error: {e}"
                )

                return (
                    f"Failed to reset the robotic arm: {e}"
                )


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

            return (
                f"Gripper open command failed. "
                f"HTTP {response.status_code}."
            )

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

            return (
                f"Gripper close command failed. "
                f"HTTP {response.status_code}."
            )

        except requests.RequestException as e:

            print(f"❌ Gripper error: {e}")
            return f"Failed to close the gripper: {e}"


    def move_joint(self, servo, angle):

        servo = int(servo)
        angle = int(angle)

        if not 0 <= servo <= 5:
            return "Invalid joint. The arm has joints 0 through 5."

        if not 0 <= angle <= 180:
            return "Invalid angle. The angle must be between 0 and 180 degrees."

        print(
            f"🤖 Moving joint {servo} "
            f"to {angle} degrees"
        )

        try:

            response = requests.post(
                f"{self.server_url}/api/servo",
                json={
                    "servo": servo,
                    "angle": angle
                },
                timeout=3,
                verify=False
            )

            if not response.ok:
                return (
                    f"Servo command failed. "
                    f"HTTP {response.status_code}."
                )

            data = response.json()

            if data.get("ok"):
                return (
                    f"Joint {servo} moved "
                    f"to {angle} degrees."
                )

            return (
                f"Joint {servo} could not be moved."
            )

        except requests.RequestException as e:

            print(f"❌ Servo API error: {e}")

            return f"Could not communicate with the robot: {e}"