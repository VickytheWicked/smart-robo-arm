import json
import re

from google import genai


class MemoryExtractor:

    def __init__(self, api_key):

        self.client = genai.Client(
            api_key=api_key
        )

    async def extract(self,conversation):

        if not conversation:
            return []

        prompt = f"""
            You are the long-term memory system for a
            robotic assistant.

            Analyze the conversation below.

            Extract only information that is useful to
            remember across future conversations.

            Good memories include:

            - User's name
            - Long-term projects
            - Current technical projects
            - Technical interests
            - Long-term goals
            - Stable preferences
            - Important robot configuration
            - Ongoing project decisions

            Do NOT store:

            - Greetings
            - Temporary questions
            - General knowledge
            - One-off facts
            - The assistant's own statements
            -Entire conversation responses

            Return ONLY valid JSON.

            Required format:

{{
    "memories": [
        {{
            "type": "user_fact",
            "key": "name",
            "value": "Vansh"
        }}
    ]
}}

Conversation:

{conversation}
"""

        try:

            response = await self.client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            text = response.text.strip()

            text = re.sub(
                r"```json",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = text.replace(
                "```",
                ""
            ).strip()

            data = json.loads(text)

            return data.get(
                "memories",
                []
            )

        except Exception as e:

            print( 
                f"❌ Memory extraction failed: {e}"
            )

            return []