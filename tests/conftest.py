from pathlib import Path

import pytest

from kbrag.models import Chunk, Passage

ONBOARDING = (
    "New employees receive a laptop on their first day. The IT team ships the laptop to the home address "
    "one week before the start date. Employees must enable disk encryption before connecting to the network."
)
VACATION = (
    "Full time employees accrue twenty days of paid vacation per year. Vacation requests are submitted "
    "through the HR portal and approved by the direct manager."
)


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "hr").mkdir(parents=True)
    (folder / "onboarding.md").write_text(ONBOARDING, encoding="utf-8")
    (folder / "hr" / "vacation.txt").write_text(VACATION, encoding="utf-8")
    (folder / "ignored.pdf").write_bytes(b"%PDF")
    return folder


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="onboarding-0000", doc_id="onboarding", source="onboarding.md", index=0, text=ONBOARDING),
        Chunk(chunk_id="vacation-0000", doc_id="vacation", source="hr/vacation.txt", index=0, text=VACATION),
    ]


@pytest.fixture
def passages(chunks: list[Chunk]) -> list[Passage]:
    return [
        Passage(text=c.text, score=0.9 - i * 0.1, source=c.source, chunk_id=c.chunk_id) for i, c in enumerate(chunks)
    ]
