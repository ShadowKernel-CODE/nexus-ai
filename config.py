import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "MemoryBot"
    APP_DESCRIPTION: str = "AI Memory Companion"

    DATABASE_URL: str = "sqlite:///./memorybot.db"

    SECRET_KEY: str = "memorybot-secret-key-change-in-production"
    SESSION_EXPIRY_HOURS: int = 24

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "openrouter/free"

    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.png,.jpg,.jpeg,.gif,.mp3,.wav,.m4a,.webm,.mp4"

    HF_TOKEN: str = ""

    ENCRYPTION_KEY: str = "encryption-key-change-in-production-32b!"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
