from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Document(_Frozen):
    source: str = Field(min_length=1)
    text: str


class Chunk(_Frozen):
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    index: int = Field(ge=0)
    text: str


class Passage(_Frozen):
    chunk_id: str
    source: str
    text: str
    score: float


class Citation(_Frozen):
    marker: int = Field(ge=1)
    source: str
    chunk_id: str


class Answer(_Frozen):
    question: str
    text: str
    citations: list[Citation]
    passages: list[Passage]
