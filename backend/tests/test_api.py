from __future__ import annotations

import json

from conftest import TODAY
from fakes import ScriptedChatModel, call_tool, structured
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.assistant import TravelAssistant
from app.graph import build_graph
from app.main import create_app


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line.startswith("data: ")]


def make_client(scripts, research_tools, planning_tools) -> TestClient:
    graph = build_graph(ScriptedChatModel(scripts), research_tools, planning_tools, today=lambda: TODAY)
    return TestClient(create_app(TravelAssistant(graph)))


def test_chat_streams_events_and_keeps_history(research_tools, planning_tools):
    scripts = {
        "supervisor": [
            structured("SupervisorDecision", reason="Needs a conversion.", next="planner", task="Convert 250 EUR to SEK.", reply=""),
            structured("SupervisorDecision", reason="Answered.", next="finish", task="", reply=""),
        ],
        "planner": [
            call_tool("convert_currency", "c1", amount=250, from_currency="EUR", to_currency="SEK"),
            AIMessage(content="250 EUR is about 2,798.70 SEK."),
        ],
    }
    with make_client(scripts, research_tools, planning_tools) as client:
        response = client.post("/api/chat", json={"thread_id": "abc", "message": "250 EUR in SEK?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert [e["type"] for e in events] == ["route", "tool_call", "tool_result", "agent_output", "route", "final"]
        assert events[-1]["content"] == "250 EUR is about 2,798.70 SEK."

        history = client.get("/api/threads/abc").json()["messages"]
        assert history == [
            {"role": "user", "content": "250 EUR in SEK?"},
            {"role": "assistant", "content": "250 EUR is about 2,798.70 SEK."},
        ]


def test_chat_reports_failures_as_an_error_event(research_tools, planning_tools):
    with make_client({"supervisor": []}, research_tools, planning_tools) as client:  # model has nothing to say
        events = parse_sse(client.post("/api/chat", json={"thread_id": "x", "message": "hi"}).text)
    assert events[-1]["type"] == "error"


def test_chat_validates_input(research_tools, planning_tools):
    with make_client({}, research_tools, planning_tools) as client:
        assert client.post("/api/chat", json={"thread_id": "x", "message": ""}).status_code == 422
