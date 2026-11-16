from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings read from RAG_* environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", frozen=True, extra="ignore")

    backend: Literal["local", "bedrock"] = "local"
    aws_region: str = "us-east-1"
    knowledge_base_id: str | None = None
    model_id: str = "amazon.nova-lite-v1:0"
    top_k: int = Field(default=3, ge=1)
    chunk_words: int = Field(default=120, ge=1)
    overlap_words: int = Field(default=20, ge=0)
