class RagError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(RagError):
    """Configuration is missing or inconsistent."""


class IngestionError(RagError):
    """Documents could not be loaded, chunked or staged."""


class RetrievalError(RagError):
    """The knowledge base could not be queried or its index could not be read."""


class GenerationError(RagError):
    """The generator failed or returned something unusable."""


class NoEvidenceError(RagError):
    """Retrieval returned nothing to ground an answer on."""


class CitationError(RagError):
    """An answer has no citations, or a citation does not map to a real source."""


class EvaluationError(RagError):
    """The evaluation inputs are missing or invalid."""
