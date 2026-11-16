from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    text: str


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source: str
    index: int = Field(ge=0)
    text: str


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    score: float
    source: str
    chunk_id: str


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1)
    source: str
    chunk_id: str
    quote: str


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    citations: list[Citation]
    passages: list[Passage]


class EvalItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)


class EvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answered: bool
    faithfulness: float
    answer_relevance: float


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: list[EvalResult]

    @property
    def mean_faithfulness(self) -> float:
        return _mean([r.faithfulness for r in self.results])

    @property
    def mean_answer_relevance(self) -> float:
        return _mean([r.answer_relevance for r in self.results])

    def passed(self, threshold: float) -> bool:
        return self.mean_faithfulness >= threshold and self.mean_answer_relevance >= threshold


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
