from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "amazon.nova-lite-v1:0"
DEFAULT_TOP_K = 5


class Settings(BaseSettings):
    """Runtime settings read from KBRAG_* environment variables or .env."""

    model_config = SettingsConfigDict(env_prefix="KBRAG_", env_file=".env", frozen=True, extra="ignore")

    aws_region: str = DEFAULT_REGION
    knowledge_base_id: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    top_k: int = DEFAULT_TOP_K
