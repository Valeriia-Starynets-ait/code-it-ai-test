class KbragError(Exception):
    """Base class for all pipeline errors."""


class IngestionError(KbragError):
    """Documents could not be loaded, chunked or written."""


class RetrievalError(KbragError):
    """The knowledge base query failed."""


class NoEvidenceError(KbragError):
    """Retrieval returned no passages to answer from."""


class CitationError(KbragError):
    """An answer is missing citations or cites an unknown source."""


class EvaluationError(KbragError):
    """The evaluation set is invalid."""


class ConfigError(KbragError):
    """Required configuration is missing."""
