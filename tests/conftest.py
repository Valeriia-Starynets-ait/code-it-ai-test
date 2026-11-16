from pathlib import Path

import pytest

from ragkb.models import Passage

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_KNOWLEDGE_BASE_ID", raising=False)


@pytest.fixture
def sample_docs() -> Path:
    return REPO_ROOT / "data" / "docs"


@pytest.fixture
def sample_questions() -> Path:
    return REPO_ROOT / "data" / "eval" / "questions.json"


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.md").write_text("Alpha systems store data in buckets. Buckets are durable.", encoding="utf-8")
    (folder / "beta.txt").write_text("Beta pipelines move events between queues.", encoding="utf-8")
    return folder


@pytest.fixture
def passages() -> list[Passage]:
    return [
        Passage(chunk_id="alpha.md#0", source="alpha.md", text="Buckets are durable. Alpha stores data.", score=0.9),
        Passage(chunk_id="beta.txt#0", source="beta.txt", text="Beta pipelines move events.", score=0.4),
    ]
