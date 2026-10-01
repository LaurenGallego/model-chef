from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from model_chef.schemas import OutputPolicy, SectionType, VenueTier


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


class RankingWeights(BaseModel):
    """Multipliers applied on top of fused relevance. See ARCHITECTURE.md §7."""

    model_config = ConfigDict(frozen=True)

    tier: dict[VenueTier, float] = {
        "peer_reviewed": 1.00,
        "official_docs": 1.00,
        "preprint": 0.90,
        "blog": 0.85,
    }
    section: dict[SectionType, float] = Field(
        default={"related_work": 0.75}, description="Sections not listed weigh 1.0."
    )
    dropped_sections: frozenset[SectionType] = frozenset({"references"})
    recency_tau_days: float = Field(default=540, description="exp(-age_days / tau).")
    recency_floor: float = Field(
        default=0.70, description="Stops strong older work being buried by recent noise."
    )
    rrf_k: int = 60
