"""
Configuration Module
Uses pydantic for type-safe configuration from environment variables
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys (for Phase 2)
    ASSEMBLYAI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    
    # Security (for Phase 2)
    JWT_SECRET_KEY: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Audio Configuration
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_DURATION_MS: int = 30
    VAD_AGGRESSIVENESS: int = 2  # 0-3 (2 = balanced)
    
    # Model Configuration
    WHISPER_MODEL: str = "medium"  # Options: tiny, base, small, medium, large
    SOURCE_LANG: str = "en"
    TARGET_LANG: str = "hi"
    
    # Performance
    MAX_AUDIO_QUEUE_SIZE: int = 100
    SILENCE_FRAMES_THRESHOLD: int = 25  # ~750ms at 30ms/frame
    MIN_SPEECH_DURATION_SECONDS: float = 0.3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a single, globally accessible settings instance
settings = Settings()