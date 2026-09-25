from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from model_chef.schemas import OutputPolicy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEL_CHEF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_url: str = Field(
        default="https://api.model-chef.dev",
        description="model-chef API endpoint the MCP server queries.",
    )
    api_key: str | None = Field(
        default=None,
        description="Key for the model-chef API. Not any LLM provider key.",
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Must match the model the index was built with. Changing it "
        "requires a full re-embed; the ingest pipeline refuses to mix models.",
    )
    embedding_dims: int = 1024

    output_policy: OutputPolicy = Field(
        default="snippet_only",
        description="'snippet_only' truncates every result. 'license_aware' serves "
        "full text for permissively licensed sources only. See docs/ARCHITECTURE.md.",
    )

    default_k: int = 5
    overfetch_factor: int = Field(
        default=3, description="Candidates fetched per result before re-ranking."
    )

    request_timeout_s: float = 20.0
