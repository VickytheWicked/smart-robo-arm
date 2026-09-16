from google.genai import types


def create_robot_tools():
    """
    Creates function declarations for Gemini Live, including
    manual movements, gripper operations, and dynamic target picking.
    """
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="pick_color_object",
                    description=(
                        "Locate, track, and pick up an object of a specific color (such as a red, green, "
                        "yellow, blue, or white lego brick, block, or eraser) using high-speed local computer vision. "
                        "ALWAYS use this tool whenever the user asks to pick, grab, or fetch an object identified by its color "
                        "(e.g. 'pick the red lego brick', 'grab the green block', 'pick up the yellow lego', 'fetch the blue brick')."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "color": types.Schema(
                                type="STRING",
                                description="The color of the object: 'red', 'green', 'yellow', 'blue', or 'white'."
                            ),
                            "item": types.Schema(
                                type="STRING",
                                description="Optional item name or description, e.g. 'lego brick', 'block', 'eraser'."
                            )
                        },
                        required=["color"]
                    )
                ),
                types.FunctionDeclaration(
                    name="pick_object",
                    description=(
                        "Locate, track, and pick up any specified target object using "
                        "the robotic arm's vision and visual servoing system. "
                        "Use this for general arbitrary objects (e.g. 'pen', 'scissors', 'marker', 'watch'). "
                        "Note: For colored objects like lego bricks or blocks, pick_color_object is preferred."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "target": types.Schema(
                                type="STRING",
                                description="The name or description of the object to locate and pick up."
                            ),
                            "color": types.Schema(
                                type="STRING",
                                description="Optional color of the object (e.g. 'red', 'green', 'yellow', 'blue', 'white') if applicable."
                            )
                        },
                        required=["target"]
                    )
                ),
                # types.FunctionDeclaration(
                #     name="stop_pick",
                #     description=(
                #         "Stop active object picking or visual servoing grab operation. "
                #         "Use when the user asks to cancel pick or halt grabbing."
                #     ),
                #     parameters=types.Schema(
                #         type="OBJECT",
                #         properties={}
                #     )
                # ),
                types.FunctionDeclaration(
                    name="stop_tracking",
                    description=(
                        "Stop active object tracking and visual servoing immediately. "
                        "Use when the user asks to stop tracking, stop detecting, "
                        "halt camera tracking, or cancel following an object."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),
                types.FunctionDeclaration(
                    name="move_arm",
                    description=(
                        "Move the robotic arm using one of the available directional commands. "
                        "Use LEFT, RIGHT, FORWARD, BACK, or CENTERED."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "command": types.Schema(
                                type="STRING",
                                description="Robot movement command: LEFT, RIGHT, FORWARD, BACK, or CENTERED."
                            )
                        },
                        required=["command"]
                    )
                ),
                types.FunctionDeclaration(
                    name="reset_arm",
                    description="Reset the robotic arm to home/idle position and clear visual servoing state.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),
                types.FunctionDeclaration(
                    name="open_gripper",
                    description="Open the robotic arm's gripper to release an object.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),
                types.FunctionDeclaration(
                    name="close_gripper",
                    description="Close the robotic arm's gripper to grab an object.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),
                types.FunctionDeclaration(
                    name="move_joint",
                    description=(
                        "Move one specific joint of the robotic arm to an angle (0 to 180 degrees). "
                        "Joint numbers are 0 (Base), 1 (Shoulder), 2 (Elbow), 3 (Wrist), 4 (WristRoll), 5 (Gripper)."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "servo": types.Schema(
                                type="INTEGER",
                                description="Joint/servo index from 0 to 5."
                            ),
                            "angle": types.Schema(
                                type="INTEGER",
                                description="Target angle in degrees from 0 to 180."
                            )
                        },
                        required=["servo", "angle"]
                    )
                ),
            ]
        )
    ]
