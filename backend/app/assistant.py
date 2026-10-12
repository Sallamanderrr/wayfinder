"""Composition root: wires settings, LLM, knowledge base, services, tools and graph together."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import HumanMessage

from .config import Settings
from .graph import build_graph, turn_input
from .rag import KnowledgeBase
from .services.attractions import AttractionsDB
from .services.currency import CurrencyService
from .services.weather import WeatherService
from .tools import build_planning_tools, build_research_tools


class TravelAssistant:
    """The interface the API uses: stream one turn of a conversation, or read its history."""

    def __init__(self, graph: Any, max_agent_runs: int = 6):
        self._graph = graph
        # Each agent run is two graph steps (agent + supervisor); leave headroom for the rest.
        self._recursion_limit = 2 * max_agent_runs + 6

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self._recursion_limit}

    def stream_turn(self, thread_id: str, message: str) -> Iterator[dict[str, Any]]:
        """Yield UI events (route, tool_call, tool_result, agent_output, review, final) as they happen."""
        yield from self._graph.stream(turn_input(message), self._config(thread_id), stream_mode="custom")

    def history(self, thread_id: str) -> list[dict[str, str]]:
        snapshot = self._graph.get_state(self._config(thread_id))
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.text}
            for m in snapshot.values.get("messages", [])
        ]


def build_assistant(settings: Settings) -> TravelAssistant:
    _check_credentials(settings)
    model_kwargs = {"temperature": settings.llm_temperature} if settings.llm_temperature is not None else {}
    llm = init_chat_model(settings.llm_model, **model_kwargs)
    embeddings = init_embeddings(settings.embedding_model)

    knowledge_base = KnowledgeBase.from_directory(settings.data_dir / "guides", embeddings)
    attractions = AttractionsDB.from_csv(settings.data_dir / "attractions.csv")
    http = httpx.Client(timeout=10.0, headers={"User-Agent": "wayfinder-demo/1.0"})

    graph = build_graph(
        llm,
        research_tools=build_research_tools(knowledge_base, attractions, WeatherService(http)),
        planning_tools=build_planning_tools(CurrencyService(http)),
        max_agent_runs=settings.max_agent_runs,
        max_reviews=settings.max_reviews,
    )
    return TravelAssistant(graph, max_agent_runs=settings.max_agent_runs)


def _check_credentials(settings: Settings) -> None:
    uses_openai = any(model.startswith("openai:") for model in (settings.llm_model, settings.embedding_model))
    if uses_openai and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy backend/.env.example to backend/.env and add your key."
        )
