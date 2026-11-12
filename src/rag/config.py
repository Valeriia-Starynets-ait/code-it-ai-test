"""Configuration via Pydantic ``BaseSettings``.

The embedding model, dimension and chunking strategy are recorded here as the
deliberate choices the bedrock-kb-rag skill requires (they cannot be changed
after a data source is created). No secrets are hardcoded; overrides come from
the environment or a ``.env`` file.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Titan Embed Text v2 supports 1024/512/256 dims; the KB fails to create if the
# configured dimension does not match the model, so this pair is chosen together.
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSION = 1024


class ChunkingStrategy(str, Enum):
    """Chunking strategy for the data source (fixed up front, immutable after).

    Uses a ``str`` mixin rather than ``enum.StrEnum`` so it works on the project's
    3.10 target (``StrEnum`` is 3.11+).
    """

    FIXED_SIZE = "FIXED_SIZE"
    HIERARCHICAL = "HIERARCHICAL"
    SEMANTIC = "SEMANTIC"


class Settings(BaseSettings):
    """Runtime settings for the RAG pipeline.

    Defaults describe the intended Bedrock Knowledge Base shape; nothing here
    performs a network call. Override via environment variables prefixed ``RAG_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        frozen=True,
        extra="ignore",
    )

    aws_region: str = "us-east-1"
    log_level: str = "INFO"

    knowledge_base_id: str = Field(default="kb-local-fake", min_length=1)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, ge=1)
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_max_tokens: int = Field(default=300, ge=1)
    chunk_overlap_percent: int = Field(default=20, ge=0, le=99)

    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
