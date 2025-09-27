from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "CreativeIQ"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]

    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "creativeiq"
    POSTGRES_PORT: int = 5432

    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "creativeiq"

    REDIS_URL: str = "redis://localhost:6379"

    # AI Model Settings
    HF_TOKEN: Optional[str] = None
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf"  # Default VLM model
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
    MODEL_CACHE_DIR: str = "./models"

    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    UPLOAD_DIR: str = "./uploads"

    # Analysis Settings
    MAX_CONCURRENT_ANALYSES: int = 5
    ANALYSIS_TIMEOUT: int = 30  # seconds

    # Performance Thresholds
    COLOR_ANALYSIS_ACCURACY: float = 0.94
    TYPOGRAPHY_ACCURACY: float = 0.91
    LAYOUT_ACCURACY: float = 0.89
    BRAND_CONSISTENCY_ACCURACY: float = 0.86

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()