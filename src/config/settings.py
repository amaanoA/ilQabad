"""Application configuration."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Environment
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True

    # Camera
    CAMERA_DEVICE_ID: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30

    # Recognition thresholds
    DETECTION_THRESHOLD: float = 0.7
    RECOGNITION_THRESHOLD: float = 0.6
    LIVENESS_THRESHOLD: float = 0.5

    # Paths
    MODELS_DIR: str = "models"
    DATA_DIR: str = "data"

    # KaamilSMS API
    KAAMILSMS_API_URL: str = ""
    KAAMILSMS_API_KEY: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
