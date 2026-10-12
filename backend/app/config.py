"""Settings from environment variables (and backend/.env, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_temperature: float | None
    embedding_model: str
    max_agent_runs: int
    max_reviews: int
    cors_origins: list[str]
    data_dir: Path


def get_settings() -> Settings:
    load_dotenv(BACKEND_DIR / ".env")
    # Low temperature keeps routing decisions consistent. Set it to an empty value for
    # models that only accept their default temperature.
    temperature = os.getenv("LLM_TEMPERATURE", "0.2").strip()
    return Settings(
        # "provider:model" strings, resolved by LangChain's init_chat_model / init_embeddings.
        llm_model=os.getenv("LLM_MODEL", "openai:gpt-4.1-mini"),
        llm_temperature=float(temperature) if temperature else None,
        embedding_model=os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small"),
        max_agent_runs=int(os.getenv("MAX_AGENT_RUNS", "6")),
        max_reviews=int(os.getenv("MAX_REVIEWS", "2")),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
        data_dir=BACKEND_DIR / "data",
    )
