from google.genai import types


def create_robot_tools():

    return [
        types.Tool(
            function_declarations=[

                types.FunctionDeclaration(
                    name="move_arm",

                    description=(
                        "Move the robotic arm using one of "
                        "the available directional commands. "
                        "Use LEFT, RIGHT, FORWARD, BACK, or "
                        "CENTERED."
                    ),

                    parameters=types.Schema(
                        type="OBJECT",

                        properties={
                            "command": types.Schema(
                                type="STRING",
                                description=(
                                    "Robot movement command. "
                                    "Must be LEFT, RIGHT, "
                                    "FORWARD, BACK, or CENTERED."
                                )
                            )
                        },

                        required=["command"]
                    )
                ),

                types.FunctionDeclaration(
                    name="reset_arm",

                    description=(
                        "Reset the robotic arm and "
                        "visual servoing state."
                    ),

                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),
                types.FunctionDeclaration(
                    name="open_gripper",

                    description=(
                        "Open the robotic arm's gripper. "
                        "Use this when the user asks to open, "
                        "release, or loosen the gripper."
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )
                ),

                types.FunctionDeclaration(
                    name="close_gripper",

                    description=(
                        "Close the robotic arm's gripper. "
                        "Use this when the user asks to close "
                        "or grip with the gripper."
                    ),

                    parameters=types.Schema(
                        type="OBJECT",
                        properties={}
                    )  
                ),
                types.FunctionDeclaration(
                    name="move_joint",

                    description=(
                        "Move one joint of the robotic arm "
                        "to a specified angle. Joint numbers "
                        "range from 0 to 5 and angles range "
                        "from 0 to 180 degrees."
                    ),

                    parameters=types.Schema(
                        type="OBJECT",

                        properties={

                            "servo": types.Schema(
                                type="INTEGER",
                                description=(
                                "Joint/servo number from 0 to 5."
                                )
                            ),

                            "angle": types.Schema(
                                type="INTEGER",
                                description=(
                                "Target angle from 0 to 180 degrees."
                                )
                            )

                        },

                        required=[
                            "servo",
                            "angle"
                        ]       
                    )
                ),
            ]
        )
    ]