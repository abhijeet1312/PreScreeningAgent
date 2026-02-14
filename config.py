"""
Configuration for Sarvam Voice Agent System
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # API Keys
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    
    # Server Configuration
    WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "localhost:8000")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Sarvam API Configuration
    SARVAM_API_HOST = os.getenv("SARVAM_API_HOST", "api.sarvam.ai")
    
    # Transcriber Settings
    TRANSCRIBER_MODEL = os.getenv("TRANSCRIBER_MODEL", "saarika:v2.5")
    TRANSCRIBER_LANGUAGE = os.getenv("TRANSCRIBER_LANGUAGE", "hi-IN")  # Hindi
    TRANSCRIBER_VAD_SENSITIVITY = os.getenv("TRANSCRIBER_VAD_SENSITIVITY", "true")
    
    # Synthesizer Settings
    SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "bulbul:v2")
    SYNTHESIZER_VOICE = os.getenv("SYNTHESIZER_VOICE", "manisha")
    SYNTHESIZER_LANGUAGE = os.getenv("SYNTHESIZER_LANGUAGE", "hi-IN")
    SYNTHESIZER_SPEED = float(os.getenv("SYNTHESIZER_SPEED", "0.8"))
    SYNTHESIZER_PITCH = float(os.getenv("SYNTHESIZER_PITCH", "0"))
    SYNTHESIZER_LOUDNESS = float(os.getenv("SYNTHESIZER_LOUDNESS", "1.0"))
    SYNTHESIZER_BUFFER_SIZE = int(os.getenv("SYNTHESIZER_BUFFER_SIZE", "100"))
    
    # Audio Settings
    TWILIO_SAMPLE_RATE = 8000  # Twilio uses 8kHz μ-law
    SARVAM_SAMPLE_RATE = 16000  # Sarvam uses 16kHz linear PCM
    AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "640"))  # bytes
    
    # Agent Settings
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "15"))
    INTERRUPTION_MIN_LENGTH = int(os.getenv("INTERRUPTION_MIN_LENGTH", "3"))
    VAD_TIMEOUT_MS = int(os.getenv("VAD_TIMEOUT_MS", "1200"))
    
    # Debug Settings
    ENABLE_TEST_TONE = os.getenv("ENABLE_TEST_TONE", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    
    # Auto-hangup config
    IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))  # idle seconds before auto-hangup
    HANGUP_PHRASES = os.getenv("HANGUP_PHRASES", "bye,goodbye,thank you,thanks,not interested,अलविदा,धन्यवाद").split(",")
    DTMF_HANGUP_KEYS = os.getenv("DTMF_HANGUP_KEYS", "#,0").split(",")
    HANGUP_MIN_LEN = int(os.getenv("HANGUP_MIN_LEN", "2"))
    CALL_RESULT_WEBHOOK_URL = os.getenv("CALL_RESULT_WEBHOOK_URL", "/call_result")
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = [
            "SARVAM_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
        ]
        
        missing = [key for key in required if not getattr(cls, key)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True


# # System prompt for the AI agent
# SYSTEM_PROMPT = """You are a friendly FEMALE sales agent from AurJobs AI helping customers understand our AI-powered recruitment platform.
# System prompt for the AI agent
SYSTEM_PROMPT = """You are a friendly FEMALE Naukri.com sales agent helping customers understand our job posting subscription plans.
RESPOND IN HINDI IF THE CUSTOMER SPEAKS IN HINDI, OTHERWISE RESPOND IN ENGLISH.

IMPORTANT:
- A greeting and introduction have ALREADY been played automatically.
- DO NOT repeat your name or re-introduce yourself.
- Start directly by asking about hiring needs.

AVAILABLE PLANS:
1. HOT VACANCY (₹1,650 + GST) - Most Popular
   - Detailed job description, 3 locations
   - Unlimited applies for 90 days
   - Job Branding, Boost on Search
   - Valid 30 days

2. CLASSIFIED (₹850 + GST) - Best Value
   - 250 character description, 3 locations
   - Unlimited applies for 90 days
   - Job Branding, Boost on Search
   - Valid 30 days

3. STANDARD (₹400 + GST) - Budget Option
   - 250 character description, 1 location
   - 200 applies for 30 days
   - Job Branding, Boost on Search
   - Valid 15 days

4. FREE POSTING
   - 250 characters, 1 location
   - 50 applies for 15 days
   - Valid 7 days
   - Not for gmail/yahoo emails

YOUR APPROACH:
1. Be friendly and conversational
2. Ask about hiring needs (role, urgency, budget)
3. Keep responses SHORT (1-2 sentences)
4. Ask ONE question at a time
5. Be helpful, not pushy
6. ₹ pronounced as 'Rupees'
7. If customer wants to end call (अलविदा, bye, धन्यवाद, thank you, hang up, not interested) - say only: "HANGUP_NOW"
"""