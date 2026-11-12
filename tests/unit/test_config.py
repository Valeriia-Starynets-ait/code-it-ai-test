from __future__ import annotations

import pytest

from rag.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    ChunkingStrategy,
    Settings,
)

pytestmark = pytest.mark.unit


def test_defaults_match_titan_embed_v2() -> None:
    settings = Settings()
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION
    assert settings.chunking_strategy is ChunkingStrategy.FIXED_SIZE


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_AWS_REGION", "eu-west-1")
    settings = Settings()
    assert settings.top_k == 9
    assert settings.aws_region == "eu-west-1"


def test_settings_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValueError):
        settings.top_k = 10  # type: ignore[misc]
