# settings.py
"""Configuration module for TempoIA API.
Uses pydantic-settings (separate package) for BaseSettings.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Authentication token – can be overridden via .env; default provided for development
    api_token: str = Field("change-me", env="API_TOKEN")
    # Path to the SQLite database used by TempoWeatherPredictor
    db_path: str = Field("tempo_weather.db", env="DB_PATH")
    # Cache TTL in seconds (default 1 hour)
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    # Optional Redis URL for a persistent cache (if not set, in‑memory cache is used)
    redis_url: str | None = Field(default=None, env="REDIS_URL")
    # Rate‑limit: max requests per minute per IP (e.g. "60/minute")
    rate_limit: str = Field("60/minute", env="RATE_LIMIT")
    # CORS origins (comma‑separated list)
    cors_origins: str = Field("*", env="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'allow'
        
# Export a singleton instance for import elsewhere
settings = Settings()
