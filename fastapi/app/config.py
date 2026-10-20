from functools import lru_cache

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: AnyHttpUrl = "http://ollama:11434"
    ollama_default_model: str = "llama3.1:8b"
    api_bearer_token: str | None = None
    ollama_connect_timeout: float = Field(5, gt=0)
    ollama_read_timeout: float = Field(120, gt=0)
    cors_origins: str = ""

    @field_validator("ollama_default_model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OLLAMA_DEFAULT_MODEL must not be blank")
        return value.strip()

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
