import asyncio
import inspect
import logging
import traceback

from google import genai
from google.genai import types

from memory.memory_manager import MemoryManager
from memory.memory_extractor import MemoryExtractor


logger = logging.getLogger(__name__)


class GeminiLive:
    """
    Handles interaction with the Gemini Live API
    and persistent long-term memory.
    """

    def __init__(self,api_key,model,input_sample_rate,tools=None,tool_mapping=None):

        self.api_key = api_key
        self.model = model
        self.input_sample_rate = (input_sample_rate)
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = (tool_mapping or {})

        # =====================================================
        # PERSISTENT MEMORY
        # =====================================================

        self.memory = MemoryManager()
        self.memory_extractor = (MemoryExtractor(api_key))
        self.session_id = None
        self.session_messages = []

        # logger.info(f"{self.memory.print_memories()}")

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def build_system_prompt(self):

        memory_context = (self.memory.get_memory_context())

        return f"""
        Your name is Lucy. You are the AI brain of a robotic assistant. The user's name is Vansh. Vansh is a computer science student and 
        robotics enthusiast who is currently building a robotic arm. You are having a natural voice conversation with him. For technical questions, be clear and practical.
        For casual conversation, be natural and friendly. Keep responses concise unless Vansh asks for a detailed explanation. Your responses will be spoken aloud. 
        When Vansh seems discouraged, stressed, or overwhelmed, be warm, patient, and empathetic. Listen without judgment and help him process what he is feeling.
        You are connected to a physical robotic arm.

        You have access to robot control tools.

        When the user explicitly asks you to control
        the robot, use the appropriate robot tool.

        Never claim that the robot moved unless the
        corresponding tool actually reports success.

        If a robot action fails, explain that it failed.

        The stop_arm tool should be treated as a high-priority safety command.
        
        If you don't know something, be honest. You have persistent memory from previous conversations. Use the persistent memory naturally when relevant. Do not mention the 
        internal memory system unless Vansh specifically asks about it.
        Persistent memory: 
            {memory_context}
        """

    # =========================================================
    # EXTRACT LONG-TERM MEMORIES
    # =========================================================

    async def extract_session_memories(self):

        if not self.session_messages:

            logger.info("No conversation messages to " 
                        "extract memories from.")
            return

        conversation_lines = []

        for role, text in self.session_messages:

            conversation_lines.append(f"{role.upper()}: {text}")

        conversation = "\n".join(conversation_lines)

        logger.info("🧠 Extracting long-term memories...")

        try:
            memories = await (self.memory_extractor.extract(conversation))
            if not memories:
                logger.info("🧠 No new memories found.")
                return
            
            saved_count = 0

            for memory in memories:

                memory_type = memory.get("type","general")

                key = memory.get("key")

                value = memory.get("value")

                if not key or not value:
                    continue

                self.memory.save_memory(memory_type,key,value)
                saved_count += 1

            logger.info(f"🧠 Saved {saved_count} " 
                        f"persistent memories.")

        except Exception as e:

            logger.error("❌ Memory extraction failed: " 
                         f"{e}\n" 
                         f"{traceback.format_exc()}")

    # =========================================================
    # START GEMINI LIVE SESSION
    # =========================================================

    async def start_session(
        self,
        audio_input_queue,
        video_input_queue,
        text_input_queue,
        audio_output_callback,
        audio_interrupt_callback=None
    ):

        # =====================================================
        # CREATE NEW MEMORY SESSION
        # =====================================================

        self.session_id = (
            self.memory.start_session()
        )

        self.session_messages = []

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = (
            self.build_system_prompt()
        )

        # =====================================================
        # GEMINI CONFIG
        # =====================================================

        config = types.LiveConnectConfig(

            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=
                        types.PrebuiltVoiceConfig(
                            voice_name="Leda"
                        )
                )
            ),

            system_instruction=types.Content(
                parts=[
                    types.Part(text=system_prompt)
                ]
            ),

            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                    turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY"
                ),
            tools=self.tools,
        )

        logger.info(
            f"Connecting to Gemini Live "
            f"with model={self.model}"
        )

        try:
            async with self.client.aio.live.connect(
                model=self.model,
                config=config

            ) as session:

                logger.info(
                    "Gemini Live session opened "
                    "successfully"
                )

                # =================================================
                # AUDIO SENDER
                # =================================================

                async def send_audio():

                    try:

                        while True:

                            chunk = (
                                await audio_input_queue.get()
                            )

                            if chunk is None:
                                break

                            await session.send_realtime_input(

                                audio=types.Blob(

                                    data=chunk,

                                    mime_type=(
                                        "audio/pcm;"
                                        f"rate={self.input_sample_rate}"
                                    )
                                )
                            )

                    except asyncio.CancelledError:

                        logger.debug(
                            "send_audio task cancelled"
                        )

                    except Exception as e:

                        logger.error(
                            f"send_audio error: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                # =================================================
                # VIDEO SENDER
                # =================================================

                async def send_video():

                    try:

                        while True:

                            chunk = (
                                await video_input_queue.get()
                            )

                            if chunk is None:

                                break

                            logger.debug(
                                "Sending video frame "
                                f"to Gemini: {len(chunk)} bytes"
                            )

                            await session.send_realtime_input(

                                video=types.Blob(

                                    data=chunk,

                                    mime_type="image/jpeg"
                                )
                            )

                    except asyncio.CancelledError:

                        logger.debug(
                            "send_video task cancelled"
                        )

                    except Exception as e:

                        logger.error(
                            f"send_video error: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                # =================================================
                # TEXT SENDER
                # =================================================

                async def send_text():

                    try:

                        while True:

                            text = (
                                await text_input_queue.get()
                            )

                            if text is None:

                                break

                            logger.info(
                                f"Sending text to Gemini: {text}"
                            )

                            # -------------------------------------
                            # SAVE TEXT MESSAGE
                            # -------------------------------------

                            self.memory.save_message(

                                self.session_id,

                                "user",

                                text
                            )

                            self.session_messages.append(
                                (
                                    "user",
                                    text
                                )
                            )

                            await session.send_realtime_input(
                                text=text
                            )

                    except asyncio.CancelledError:

                        logger.debug(
                            "send_text task cancelled"
                        )

                    except Exception as e:

                        logger.error(
                            f"send_text error: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                # =================================================
                # EVENT QUEUE
                # =================================================

                event_queue = (
                    asyncio.Queue()
                )

                # =================================================
                # RECEIVE LOOP
                # =================================================

                async def receive_loop():

                    try:

                        while True:

                            async for response in (
                                session.receive()
                            ):

                                logger.debug(
                                    "Received response "
                                    "from Gemini"
                                )

                                # ---------------------------------
                                # GO AWAY
                                # ---------------------------------

                                if response.go_away:

                                    logger.warning(
                                        f"Received GoAway: "
                                        f"{response.go_away}"
                                    )

                                # ---------------------------------
                                # SESSION RESUMPTION
                                # ---------------------------------

                                if (
                                    response
                                    .session_resumption_update
                                ):

                                    logger.info(
                                        "Session resumption update: "
                                        f"{response.session_resumption_update}"
                                    )

                                server_content = (
                                    response.server_content
                                )

                                tool_call = (
                                    response.tool_call
                                )

                                # =================================
                                # SERVER CONTENT
                                # =================================

                                if server_content:

                                    # -----------------------------
                                    # AUDIO
                                    # -----------------------------

                                    if (
                                        server_content
                                        .model_turn
                                    ):

                                        for part in (
                                            server_content
                                            .model_turn
                                            .parts
                                        ):

                                            if (
                                                part.inline_data
                                            ):

                                                audio_data = (
                                                    part
                                                    .inline_data
                                                    .data
                                                )

                                                if (
                                                    inspect
                                                    .iscoroutinefunction(
                                                        audio_output_callback
                                                    )
                                                ):

                                                    await (
                                                        audio_output_callback(
                                                            audio_data
                                                        )
                                                    )

                                                else:

                                                    audio_output_callback(
                                                        audio_data
                                                    )

                                    # -----------------------------
                                    # USER TRANSCRIPTION
                                    # -----------------------------

                                    if (
                                        server_content
                                        .input_transcription
                                        and
                                        server_content
                                        .input_transcription
                                        .text
                                    ):

                                        user_text = (
                                            server_content
                                            .input_transcription
                                            .text
                                        )

                                        logger.info(
                                            f"User: "
                                            f"{user_text}"
                                        )

                                        # Save to SQLite
                                        self.memory.save_message(

                                            self.session_id,

                                            "user",

                                            user_text
                                        )

                                        # Save to current session
                                        self.session_messages.append(
                                            (
                                                "user",
                                                user_text
                                            )
                                        )

                                        await event_queue.put({

                                            "type":
                                                "user",

                                            "text":
                                                user_text

                                        })

                                    # -----------------------------
                                    # GEMINI TRANSCRIPTION
                                    # -----------------------------

                                    if (
                                        server_content
                                        .output_transcription
                                        and
                                        server_content
                                        .output_transcription
                                        .text
                                    ):

                                        gemini_text = (
                                            server_content
                                            .output_transcription
                                            .text
                                        )

                                        logger.info(
                                            f"Gemini: "
                                            f"{gemini_text}"
                                        )

                                        # Save to SQLite
                                        self.memory.save_message(

                                            self.session_id,

                                            "assistant",

                                            gemini_text
                                        )

                                        # Save to current session
                                        self.session_messages.append(
                                            (
                                                "assistant",
                                                gemini_text
                                            )
                                        )

                                        await event_queue.put({

                                            "type":
                                                "gemini",

                                            "text":
                                                gemini_text

                                        })

                                    # -----------------------------
                                    # TURN COMPLETE
                                    # -----------------------------

                                    if (
                                        server_content
                                        .turn_complete
                                    ):

                                        await event_queue.put({

                                            "type":
                                                "turn_complete"

                                        })

                                    # -----------------------------
                                    # INTERRUPTION
                                    # -----------------------------

                                    if (
                                        server_content
                                        .interrupted
                                    ):

                                        logger.info(
                                            "Gemini response "
                                            "interrupted"
                                        )

                                        if (
                                            audio_interrupt_callback
                                        ):

                                            if (
                                                inspect
                                                .iscoroutinefunction(
                                                    audio_interrupt_callback
                                                )
                                            ):

                                                await (
                                                    audio_interrupt_callback()
                                                )

                                            else:

                                                audio_interrupt_callback()

                                        await event_queue.put({

                                            "type":
                                                "interrupted"

                                        })

                                # =================================
                                # TOOL CALL
                                # =================================

                                if tool_call:

                                    function_responses = []

                                    for fc in (
                                        tool_call.function_calls
                                    ):

                                        func_name = (
                                            fc.name
                                        )

                                        args = (
                                            fc.args or {}
                                        )

                                        if (
                                            func_name
                                            not in
                                            self.tool_mapping
                                        ):

                                            logger.warning(
                                                f"Unknown tool: "
                                                f"{func_name}"
                                            )

                                            continue

                                        try:

                                            tool_func = (
                                                self.tool_mapping[
                                                    func_name
                                                ]
                                            )

                                            if (
                                                inspect
                                                .iscoroutinefunction(
                                                    tool_func
                                                )
                                            ):

                                                result = (
                                                    await tool_func(
                                                        **args
                                                    )
                                                )

                                            else:

                                                loop = (
                                                    asyncio
                                                    .get_running_loop()
                                                )

                                                result = (
                                                    await loop
                                                    .run_in_executor(
                                                        None,
                                                        lambda:
                                                        tool_func(
                                                            **args
                                                        )
                                                    )
                                                )

                                        except Exception as e:

                                            logger.error(
                                                f"Tool "
                                                f"{func_name} "
                                                f"failed: {e}"
                                            )

                                            result = (
                                                f"Error: {e}"
                                            )

                                        function_responses.append(

                                            types.FunctionResponse(

                                                name=func_name,

                                                id=fc.id,

                                                response={
                                                    "result": result
                                                }
                                            )
                                        )

                                        await event_queue.put({

                                            "type":
                                                "tool_call",

                                            "name":
                                                func_name,

                                            "args":
                                                args,

                                            "result":
                                                result

                                        })

                                    if function_responses:

                                        await (
                                            session
                                            .send_tool_response(
                                                function_responses=
                                                    function_responses
                                            )
                                        )

                        # session.receive() ended
                        # Re-enter to continue listening.

                        logger.debug(
                            "Gemini receive iterator "
                            "completed, re-entering"
                        )

                    except asyncio.CancelledError:

                        logger.debug(
                            "receive_loop task cancelled"
                        )

                    except Exception as e:

                        logger.error(
                            f"receive_loop error: "
                            f"{type(e).__name__}: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                        await event_queue.put({

                            "type":
                                "error",

                            "error":
                                f"{type(e).__name__}: {e}"

                        })

                    finally:

                        logger.info(
                            "receive_loop exiting"
                        )

                        await event_queue.put(
                            None
                        )

                # =================================================
                # CREATE TASKS
                # =================================================

                send_audio_task = (
                    asyncio.create_task(
                        send_audio()
                    )
                )

                send_video_task = (
                    asyncio.create_task(
                        send_video()
                    )
                )

                send_text_task = (
                    asyncio.create_task(
                        send_text()
                    )
                )

                receive_task = (
                    asyncio.create_task(
                        receive_loop()
                    )
                )

                # =================================================
                # EVENT PROCESSING
                # =================================================

                try:

                    while True:

                        event = (
                            await event_queue.get()
                        )

                        if event is None:

                            break

                        if (
                            isinstance(
                                event,
                                dict
                            )
                            and
                            event.get("type")
                            == "error"
                        ):

                            yield event

                            break

                        yield event

                finally:

                    logger.info(
                        "Cleaning up Gemini Live "
                        "session tasks"
                    )

                    send_audio_task.cancel()

                    send_video_task.cancel()

                    send_text_task.cancel()

                    receive_task.cancel()

                    await asyncio.gather(

                        send_audio_task,
                        send_video_task,
                        send_text_task,
                        receive_task,

                        return_exceptions=True
                    )

        except Exception as e:

            logger.error(
                f"Gemini Live session error: "
                f"{type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

            raise

        finally:

            # =====================================================
            # EXTRACT LONG-TERM MEMORIES
            # =====================================================

            logger.info(
                "Gemini Live session ending. "
                "Processing persistent memory..."
            )

            try:

                await (
                    self.extract_session_memories()
                )

            except Exception as e:

                logger.error(
                    f"Failed to extract memories: "
                    f"{e}\n"
                    f"{traceback.format_exc()}"
                )

            # =====================================================
            # SAVE SESSION
            # =====================================================

            try:

                message_count = len(
                    self.session_messages
                )

                summary = (
                    f"Session contained "
                    f"{message_count} conversation "
                    f"messages."
                )

                self.memory.end_session(
                    self.session_id,
                    summary
                )

            except Exception as e:

                logger.error(
                    f"Failed to close memory session: "
                    f"{e}"
                )

            logger.info(
                "Gemini Live session closed"
            )