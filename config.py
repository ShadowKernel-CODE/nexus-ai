import os
import re

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PLACEHOLDER_PATTERNS = (
    re.compile(r"change[-_ ]this", re.IGNORECASE),
    re.compile(r"change[-_ ]me", re.IGNORECASE),
    re.compile(r"your[-_ ]", re.IGNORECASE),
    re.compile(r"placeholder", re.IGNORECASE),
    re.compile(r"^secret$", re.IGNORECASE),
    re.compile(r"memorybot[-_ ]secret", re.IGNORECASE),
)


class Settings(BaseSettings):
    APP_NAME: str = "MemoryBot"
    APP_DESCRIPTION: str = "AI Memory Companion"
    APP_ENV: str = "development"

    DATABASE_URL: str = "sqlite:///./memorybot.db"

    SECRET_KEY: str
    SESSION_EXPIRY_HOURS: int = 24

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "openrouter/free"
    VISION_MODEL: str = "gpt-4.1-mini"

    ELEVENLABS_API_KEY: str = ""

    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_TTS_VOICE: str = "aura-2-thalia-en"

    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD: str = ""

    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.png,.jpg,.jpeg,.gif,.mp3,.wav,.m4a,.webm,.mp4"
    ALLOWED_IMAGE_EXTENSIONS: str = ".png,.jpg,.jpeg,.webp"
    ALLOWED_AUDIO_EXTENSIONS: str = ".mp3,.wav,.m4a,.webm"
    ALLOWED_VIDEO_EXTENSIONS: str = ".mp4,.webm,.mov"

    HF_TOKEN: str = ""

    ENCRYPTION_KEY: str

    @property
    def cookie_secure(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @field_validator("SECRET_KEY", "ENCRYPTION_KEY")
    @classmethod
    def validate_secret(cls, value: str, info) -> str:
        name = info.field_name
        if not value or not value.strip():
            raise ValueError(f"{name} must be set in environment variables")
        if cls.is_production_secret_placeholder(value):
            raise ValueError(
                f"{name} must not be a known placeholder value in production. "
                "Generate a strong random value (e.g. python -c 'import secrets; print(secrets.token_urlsafe(48))')."
            )
        return value

    @classmethod
    def is_production_secret_placeholder(cls, value: str) -> bool:
        return any(p.search(value) for p in PLACEHOLDER_PATTERNS)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
