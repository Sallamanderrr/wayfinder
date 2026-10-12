"""Test doubles: a lexical embedding model and a scripted, role-aware chat model.

They let the whole graph run offline and deterministically -- no API key, no network.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing vectoriser: similar wording -> similar vectors."""

    def __init__(self, size: int = 512):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in re.findall(r"[a-zà-ÿ]{4,}", text.lower()):  # skip short stopwords
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.size
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


ROLES = ("supervisor", "researcher", "planner", "verifier")


class ScriptedChatModel(GenericFakeChatModel):
    """Returns pre-scripted messages, picking the script by the agent's system prompt.

    Each role has its own queue, so a test reads like the conversation it simulates:
    script["researcher"] = [AIMessage(tool_calls=[...]), AIMessage("findings")].
    Every prompt the model receives is recorded in `calls` for assertions.
    """

    messages: Any = None  # replaced before each call; required by the parent class
    scripts: Any = None
    calls: Any = None

    def __init__(self, scripts: dict[str, Iterable[AIMessage]]):
        super().__init__(messages=iter(()))
        self.scripts = defaultdict(deque, {role: deque(items) for role, items in scripts.items()})
        self.calls = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self  # tools are executed by the real ToolNode; the script decides the calls

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        role = _role_of(messages)
        self.calls.append((role, messages))
        if not self.scripts[role]:
            raise AssertionError(f"No scripted response left for the {role}")
        self.messages = iter([self.scripts[role].popleft()])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _role_of(messages: list[BaseMessage]) -> str:
    system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    text = system.text.lower() if system else ""
    for role in ROLES:
        if f"you are the {role}" in text:
            return role
    raise AssertionError(f"Could not tell which agent is calling from the system prompt: {text[:80]!r}")


def structured(name: str, **args: Any) -> AIMessage:
    """A scripted structured-output reply (sent as a tool call, like real providers do)."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def call_tool(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
