"""
Voice Agent Orchestrator (Dynamic Workflow)
Manages conversation flow, interruption handling, and component integration.

NO hardcoded workflow types — behavior is driven entirely by:
  • call_purpose  (e.g. "sales call", "interview", "follow-up")
  • context_text  (resume, product info, script, PDF content …)

The LLM system prompt is built dynamically from these two fields, so the
same voice agent can conduct a hiring interview, a sales pitch, a customer
follow-up, or anything else without code changes.
"""

import asyncio
import json
import logging
import time
import requests as http
from typing import Optional, Dict, Any, List

from openai import AsyncAzureOpenAI
from config import Config, SYSTEM_PROMPT
from sarvam_transcriber import SarvamTranscriber
from sarvam_synthesizer import SarvamSynthesizer
from audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicPromptBuilder:
    """
    Builds a system prompt and opening message dynamically based on
    call_purpose and context_text — no fixed workflow types.
    """

    # Keywords used to *infer* the conversational style from call_purpose
    STYLE_HINTS = {
        "interview": {
            "keywords": ["interview", "hiring", "screening", "assessment", "evaluate candidate"],
            "role": "an AI interview agent",
            "tone": "professional and structured",
            "goal": "assess the candidate's fit based on their background and the role requirements",
            "closing": "Thank you for your time. An HR representative will be in touch soon.",
            "max_exchanges": 12,
        },
        "sales": {
            "keywords": ["sales", "pitch", "sell", "product demo", "lead generation", "cold call"],
            "role": "a friendly AI sales representative",
            "tone": "warm, conversational, and consultative",
            "goal": "understand the customer's needs, explain the value proposition, and schedule a follow-up or send details",
            "closing": "Thank you for your time! I'll send you more details shortly.",
            "max_exchanges": 15,
        },
        "follow_up": {
            "keywords": ["follow up", "follow-up", "check in", "check-in", "update"],
            "role": "an AI assistant following up",
            "tone": "polite and brief",
            "goal": "check in on the previous conversation and see if they need anything",
            "closing": "Thank you! We'll keep you updated.",
            "max_exchanges": 10,
        },
        "survey": {
            "keywords": ["survey", "feedback", "poll", "questionnaire"],
            "role": "an AI survey agent",
            "tone": "neutral, patient, and clear",
            "goal": "collect structured feedback or answers",
            "closing": "Thank you for your feedback! We appreciate your time.",
            "max_exchanges": 15,
        },
        "support": {
            "keywords": ["support", "help desk", "troubleshoot", "complaint", "issue"],
            "role": "an AI customer support agent",
            "tone": "empathetic, calm, and solution-oriented",
            "goal": "understand the issue and provide resolution or escalate appropriately",
            "closing": "Thank you for reaching out. We'll make sure this is resolved.",
            "max_exchanges": 15,
        },
    }

    DEFAULT_STYLE = {
        "role": "an AI assistant",
        "tone": "friendly and professional",
        "goal": "have a productive conversation based on the given purpose",
        "closing": "Thank you for your time! Have a great day.",
        "max_exchanges": 12,
    }

    @classmethod
    def detect_style(cls, purpose: str) -> dict:
        """Detect conversational style from purpose text."""
        purpose_lower = (purpose or "").lower()
        for style in cls.STYLE_HINTS.values():
            if any(kw in purpose_lower for kw in style["keywords"]):
                return style
        return cls.DEFAULT_STYLE

    @classmethod
    def build_system_prompt(cls, purpose: str, context: str, extra_data: dict = None) -> str:
        """
        Build a complete system prompt dynamically.

        Args:
            purpose:    e.g. "sales call", "technical interview", "follow-up"
            context:    e.g. resume text, product info, script, PDF content
            extra_data: optional dict with keys like lead_name, company_name, etc.
        """
        style = cls.detect_style(purpose)
        extra = extra_data or {}

        # Build context section
        context_section = ""
        if context and context.strip():
            context_section = f"""
CONTEXT INFORMATION (use this to guide the conversation):
{context.strip()}
"""

        # Build extra info section (lead name, company, etc.)
        extra_section = ""
        if extra:
            lines = []
            for key, value in extra.items():
                if value and str(value).strip():
                    label = key.replace("_", " ").title()
                    lines.append(f"- {label}: {value}")
            if lines:
                extra_section = "\nADDITIONAL INFORMATION:\n" + "\n".join(lines)

        prompt = f"""You are a female voice assistant with  {style['role']}  .
        
        Pronounciation guidance:
       
        pronounce "." as "dot" when reading out URLs or email addresses.

CALL PURPOSE: {purpose or 'general conversation'}

YOUR TONE: {style['tone']}
YOUR GOAL: {style['goal']}
{context_section}{extra_section}
CONVERSATION RULES:
1. Ask ONE question at a time — keep it short and natural.
2. Listen carefully and ask relevant follow-up questions.
3. Be conversational — this is a phone call, not a written document.
4. Stay on topic. If the person goes off-track, gently redirect.
5. Maximum {style['max_exchanges']} exchanges before wrapping up.
6. End the call politely when the conversation naturally concludes or the limit is reached.

CRITICAL RULES:
- If the person wants to end the call (says bye, thanks, not interested, hang up, alvida, dhanyavaad, or similar) — respond ONLY with: "HANGUP_NOW"
- Never pressure or be pushy.
- Keep all responses under 2-3 sentences for a natural phone experience.

CLOSING LINE (when wrapping up): "{style['closing']}"
"""
        return prompt

    @classmethod
    def build_first_message(cls, purpose: str, extra_data: dict = None) -> str:
        """Generate the opening line for the call."""
        style = cls.detect_style(purpose)
        extra = extra_data or {}
        purpose_lower = (purpose or "").lower()

        # Try to use lead/candidate name if available
        name = extra.get("lead_name") or extra.get("candidate_name") or ""
        greeting_name = f" {name}" if name else ""

        company = extra.get("company_name") or extra.get("lead_company") 

        if any(kw in purpose_lower for kw in ["interview", "hiring", "screening"]):
            return f"Hello{greeting_name}! I'm an AI assistant calling from {company}. We'd like to have a quick conversation about the role you applied for. Do you have a few minutes?"

        if any(kw in purpose_lower for kw in ["sales", "pitch", "product"]):
            return f"Hello{greeting_name}! I'm calling from {company}. How are you today? I wanted to take just a moment of your time to discuss something that might be valuable for you."

        if any(kw in purpose_lower for kw in ["follow up", "follow-up", "check in"]):
            return f"Hi{greeting_name}! I'm calling from {company} to follow up on our previous conversation. Is this a good time?"

        if any(kw in purpose_lower for kw in ["survey", "feedback"]):
            return f"Hello{greeting_name}! I'm calling from {company}. We'd love to get your quick feedback. Do you have a couple of minutes?"

        if any(kw in purpose_lower for kw in ["support", "help", "issue"]):
            return f"Hello{greeting_name}! I'm calling from {company} regarding your recent inquiry. How can I help you today?"

        # Generic fallback
        return f"Hello{greeting_name}! I'm calling from {company} regarding {purpose or 'a quick conversation'}. Do you have a moment?"

    @classmethod
    def get_max_exchanges(cls, purpose: str) -> int:
        """Return the max number of exchanges for this purpose."""
        style = cls.detect_style(purpose)
        return style.get("max_exchanges", 12)

    @classmethod
    def get_closing_message(cls, purpose: str) -> str:
        """Return the closing message for this purpose."""
        style = cls.detect_style(purpose)
        return style.get("closing", "Thank you for your time!")


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceAgent:
    """
    Main voice agent orchestrator — fully dynamic, no hardcoded workflows.
    
    Behavior is driven by:
      • call_purpose  → determines conversational style & limits
      • context_text  → fed into the system prompt as background info
      • extra_data    → optional structured data (lead name, company, etc.)
    """

    def __init__(self, call_sid: str, stream_sid: str, websocket,
                 workflow_type: str, workflow_data: dict):
        """
        Initialize voice agent.

        Args:
            call_sid:      Twilio call SID
            stream_sid:    Twilio stream SID
            websocket:     FastAPI WebSocket connection
            workflow_type:  Legacy param — kept for API compat but NOT used for prompt logic
            workflow_data:  Dict with call_purpose, context_text, and optional extra fields
        """
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.ws = websocket

        # ── Dynamic workflow data ──
        self.workflow_data = workflow_data
        self.call_purpose = workflow_data.get("call_purpose", "")
        self.context_text = workflow_data.get("context_text", "")
        self.workflow_run_id = workflow_data.get("workflow_run_id")
        self.chat_id = workflow_data.get("chat_id")

        # Extra structured data (lead_name, company_name, candidate_name, etc.)
        self.extra_data = workflow_data.get("extra_data", {})

        # Legacy compat — store but don't use for prompt logic
        self._legacy_workflow_type = workflow_type

        # ── Conversation state ──
        self.conversation: List[Dict[str, str]] = []
        self.question_number = 0
        self.max_exchanges = DynamicPromptBuilder.get_max_exchanges(self.call_purpose)
        self.transcript = []

        # ── Components ──
        self.transcriber: Optional[SarvamTranscriber] = None
        self.synthesizer: Optional[SarvamSynthesizer] = None
        self.audio_processor = AudioProcessor()

        # ── LLM client ──
        self.openai_client = AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )

        # ── State flags ──
        self.is_speaking = False
        self.awaiting_response = False
        self.user_is_speaking = False
        self.conversation_ended = False

        # ── Tasks ──
        self.transcription_handler_task: Optional[asyncio.Task] = None
        self.synthesis_handler_task: Optional[asyncio.Task] = None

        # ── Performance tracking ──
        self.call_start_time = time.time()
        self.total_transcripts = 0
        self.total_responses = 0

        # ── Idle detection ──
        self.last_activity = time.time()
        self.idle_timeout = Config.IDLE_TIMEOUT_SECONDS
        self.idle_task: Optional[asyncio.Task] = None
        self.auto_hangup_enabled = True

        # ── Turn management ──
        self.processing_turn = False
        self.last_turn_id = None
        self.last_response_time = 0
        self.webhook_sent = False
        self.latest_user_utterance = None
        self.user_speaking = False
        self.response_task: Optional[asyncio.Task] = None

        # ── Disinterest tracking (for sales-style calls) ──
        self.disinterest_count = 0

        logger.info(f"workflow_data-----------------------------: {self.workflow_data}")
        logger.info(f"Call purpose: {self.call_purpose}")
        logger.info(f"Context text: {self.context_text[:200] if self.context_text else 'None'}...")
        logger.info(f"Max exchanges: {self.max_exchanges}")
        logger.info(f"Agent initialized for call {call_sid}")

    # ─────────────────────────────────────────────────────────────────────────
    # DYNAMIC PROMPT (replaces all old _load_system_prompt branches)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        """Build system prompt dynamically from call_purpose + context_text."""
        return DynamicPromptBuilder.build_system_prompt(
            purpose=self.call_purpose,
            context=self.context_text,
            extra_data=self.extra_data,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self):
        """Initialize transcriber and synthesizer."""
        try:
            self.transcriber = SarvamTranscriber()
            await self.transcriber.start()
            logger.info("Transcriber initialized")

            self.synthesizer = SarvamSynthesizer()
            await self.synthesizer.start()
            logger.info("Synthesizer initialized")

            self.transcription_handler_task = asyncio.create_task(self._handle_transcriptions())
            self.synthesis_handler_task = asyncio.create_task(self._handle_synthesis())
            self.idle_task = asyncio.create_task(self._monitor_idle_timeout())

            logger.info("Agent fully initialized")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    async def start_conversation(self):
        """Start conversation with a dynamic greeting."""
        logger.info("Starting conversation")

        first_message = DynamicPromptBuilder.build_first_message(
            purpose=self.call_purpose,
            extra_data=self.extra_data,
        )

        await self.speak(first_message)
        self.conversation.append({"role": "assistant", "content": first_message})
        self.awaiting_response = True

    async def end_call(self):
        """End call gracefully with purpose-appropriate closing."""
        logger.info("Ending call")
        self.conversation_ended = True

        closing = DynamicPromptBuilder.get_closing_message(self.call_purpose)
        await self.speak(closing)
        await asyncio.sleep(3.0)
        await self._hangup_twilio()
        await self.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIO
    # ─────────────────────────────────────────────────────────────────────────

    async def process_audio(self, audio_payload: str):
        """Process incoming audio from Twilio."""
        if self.transcriber and not self.conversation_ended:
            import base64
            audio_bytes = base64.b64decode(audio_payload)
            await self.transcriber.send_audio(audio_bytes)

    async def speak(self, text: str):
        """Speak text via synthesizer."""
        try:
            self.is_speaking = True
            self.last_activity = time.time()
            await self.synthesizer.synthesize(text, flush=True)
            logger.info(f"Speaking: {text[:50]}...")
        except Exception as e:
            logger.error(f"Speech error: {e}")
            self.is_speaking = False

    async def _stream_audio_to_twilio(self, audio_data: bytes):
        """Stream audio to Twilio."""
        try:
            import base64
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": audio_b64},
            }
            await self.ws.send_json(message)
        except Exception as e:
            logger.error(f"Twilio stream error: {e}")

    async def _send_twilio_clear(self):
        """Send clear command to Twilio to stop current audio playback."""
        try:
            await self.ws.send_json({"event": "clear", "streamSid": self.stream_sid})
            logger.debug("Clear command sent to Twilio")
        except Exception as e:
            logger.error(f"Clear command error: {e}")

    async def _hangup_twilio(self):
        """Terminate Twilio call."""
        try:
            from twilio.rest import Client
            client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
            client.calls(self.call_sid).update(status="completed")
            logger.info(f"Call {self.call_sid} terminated")
        except Exception as e:
            logger.error(f"Hangup error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSCRIPTION HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_transcriptions(self):
        """Handle incoming transcriptions and VAD events."""
        try:
            async for event in self.transcriber.transcripts():
                if self.conversation_ended:
                    break

                event_type = event.get("type")

                if event_type == "transcript":
                    text = event.get("text", "").strip()
                    is_final = event.get("is_final", False)

                    if not text:
                        continue
                    if not is_final:
                        logger.debug(f"(partial) {text}")
                        continue

                    logger.info(f"User: {text}")
                    self.total_transcripts += 1
                    self.user_is_speaking = False
                    self.last_activity = time.time()

                    # Handle interruption
                    if self.is_speaking and len(text) >= Config.INTERRUPTION_MIN_LENGTH:
                        logger.warning("User interrupted bot")
                        await self._handle_interruption()

                    self.conversation.append({"role": "user", "content": text})
                    self.transcript.append({"speaker": "candidate", "text": text})

                    if self.processing_turn:
                        logger.debug("Still processing previous turn, skipping")
                        continue

                    if self.awaiting_response:
                        self.awaiting_response = False
                        if self.response_task and not self.response_task.done():
                            self.response_task.cancel()
                        self.response_task = asyncio.create_task(self._generate_response())

                elif event_type == "vad":
                    signal = event.get("signal")
                    if signal == "START_SPEECH":
                        self.user_is_speaking = True
                    elif signal == "END_SPEECH":
                        self.user_is_speaking = False

        except Exception as e:
            logger.error(f"Transcription handler error: {e}")

    async def _handle_synthesis(self):
        """Stream synthesized audio to Twilio."""
        try:
            FRAME_DURATION = 0.02
            async for audio_chunk in self.synthesizer.audio_stream():
                if self.conversation_ended:
                    break
                await self._stream_audio_to_twilio(audio_chunk)
                await asyncio.sleep(FRAME_DURATION)
        except Exception as e:
            logger.error(f"Synthesis handler error: {e}")

    async def _handle_interruption(self):
        """Handle user interruption of bot speech."""
        try:
            self.is_speaking = False
            if self.response_task and not self.response_task.done():
                self.response_task.cancel()
                self.response_task = None
            if self.synthesizer:
                await self.synthesizer.interrupt()
            await self._send_twilio_clear()
            logger.info("Interruption handled")
        except Exception as e:
            logger.error(f"Interruption handling error: {e}")

    async def _monitor_idle_timeout(self):
        """Monitor for idle timeout and auto-hangup."""
        try:
            while not self.conversation_ended and self.auto_hangup_enabled:
                await asyncio.sleep(5)
                idle_duration = time.time() - self.last_activity
                if idle_duration >= self.idle_timeout:
                    logger.warning(f"Idle timeout reached ({idle_duration:.1f}s). Auto-hanging up.")
                    await self.end_call()
                    break
        except asyncio.CancelledError:
            logger.info("Idle monitor task cancelled")
        except Exception as e:
            logger.error(f"Idle monitor error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_response(self):
        """Generate AI response using Azure OpenAI."""
        try:
            logger.info("Generating response...")

            messages = [
                {"role": "system", "content": self._load_system_prompt()}
            ] + self.conversation

            response_text = ""
            stream = await self.openai_client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=150,
            )

            async for chunk in stream:
                if self.conversation_ended:
                    break
                if not getattr(chunk, "choices", None) or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    response_text += content_piece

            if not response_text:
                logger.warning("Empty response from LLM")
                return

            logger.info(f"Assistant: {response_text}")
            self.transcript.append({"speaker": "assistant", "text": response_text})

            # Check for hangup signal
            if "HANGUP_NOW" in response_text:
                logger.info("LLM detected hangup intent, ending call")
                await self.end_call()
                return

            # Add to conversation
            self.conversation.append({"role": "assistant", "content": response_text})
            self.total_responses += 1
            self.question_number += 1

            # ── Dynamic auto-stop logic ──
            # Track disinterest for any call purpose
            lower_response = response_text.lower()
            lower_user = (self.conversation[-2]["content"].lower()
                          if len(self.conversation) >= 2 else "")

            disinterest_phrases = [
                "not interested", "no thanks", "no thank you",
                "don't want", "don't need", "nahi chahiye", "mujhe nahi",
            ]
            if any(phrase in lower_user for phrase in disinterest_phrases):
                self.disinterest_count += 1
                logger.info(f"Disinterest detected ({self.disinterest_count})")

            # Auto-end conditions
            should_end = False
            if self.question_number >= self.max_exchanges:
                should_end = True
                logger.info(f"Max exchanges ({self.max_exchanges}) reached")
            elif self.disinterest_count >= 2:
                should_end = True
                logger.info("Disinterest threshold reached")

            if should_end:
                closing = DynamicPromptBuilder.get_closing_message(self.call_purpose)
                await self.speak(closing)
                self.conversation.append({"role": "assistant", "content": closing})
                await self.end_call()
                return

            # Speak the response
            await self.speak(response_text)
            self.awaiting_response = True

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            import traceback
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────────

    async def cleanup(self):
        """Send transcript webhook and clean up resources."""
        payload = {
            "call_sid": self.call_sid,
            "workflow_run_id": self.workflow_run_id,
            "chat_id": self.chat_id,
            "transcript": self.transcript,
        }

        logging.info(f"Webhook payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        def send_webhook():
            resp = http.post(
                Config.CALL_RESULT_WEBHOOK_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15,
            )
            return resp.status_code, resp.text

        try:
            status, text = await asyncio.to_thread(send_webhook)
            logging.info(f"Webhook response: {status} {text}")
        except Exception as e:
            logging.error(f"Webhook failed: {e}")