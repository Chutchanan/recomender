# config.py - Configuration management
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database (matching your setup)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/restaurant_recommendation"
    DATABASE_ECHO: bool = False  # Set to True for SQL logging
    
    # Model
    MODEL_PATH: str = "data/model.pt"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Performance
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()