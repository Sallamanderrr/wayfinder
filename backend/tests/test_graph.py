"""End-to-end tests of the orchestration, with a scripted LLM and offline tools.

The scripts play the part of the model; everything else (routing, tool execution, state,
guardrails, checkpointing, event streaming) is the real code.
"""

from __future__ import annotations

import pytest
from conftest import TODAY
from fakes import ScriptedChatModel, call_tool, structured
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import SupervisorDecision, WorkItem, apply_guardrails, build_graph, turn_input

USER_REQUEST = (
    "Plan 2 days in Lisbon on 2026-10-17 and 2026-10-18 for 2 people. "
    "Our budget for sights is 1000 SEK. Indoor stuff if it rains, please."
)


def route(next_: str, task: str = "", reply: str = "", reason: str = "test") -> AIMessage:
    return structured("SupervisorDecision", reason=reason, next=next_, task=task, reply=reply)


def run_turn(graph, thread: str, message: str) -> list[dict]:
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 30}
    return list(graph.stream(turn_input(message), config, stream_mode="custom"))


def make_graph(scripts, research_tools, planning_tools, **kwargs):
    llm = ScriptedChatModel(scripts)
    return llm, build_graph(llm, research_tools, planning_tools, today=lambda: TODAY, **kwargs)


def test_plan_goes_through_research_planning_review_and_revision(research_tools, planning_tools):
    llm, graph = make_graph(
        {
            "supervisor": [
                route("researcher", "Get the Lisbon forecast for 2026-10-17..18 and indoor attractions."),
                route("planner", "Build the 2-day plan for 2 people within 1000 SEK."),
                route("verifier", "Check the plan."),
                route("planner", "Fix the issue: the total leaves out the tram fares."),
                route("verifier", "Re-check the plan."),
                route("finish"),
            ],
            "researcher": [
                AIMessage(
                    content="",
                    tool_calls=[  # two tools in one step, as models do for independent lookups
                        {"name": "get_weather_forecast", "args": {"city": "Lisbon", "start_date": "2026-10-17", "end_date": "2026-10-18"}, "id": "w1"},
                        {"name": "find_attractions", "args": {"city": "Lisbon", "indoor": True, "max_price": 15}, "id": "a1"},
                    ],
                ),
                AIMessage(content="Saturday is rainy, Sunday is dry. Indoor options: Tile Museum (8 EUR)..."),
            ],
            "planner": [
                call_tool("convert_currency", "c1", amount=1000, from_currency="SEK", to_currency="EUR"),
                call_tool("calculator", "k1", expression="2 * (8 + 14)"),
                AIMessage(content="Draft v1: Tile Museum + Gulbenkian, total 44 EUR."),
                call_tool("calculator", "k2", expression="2 * (8 + 14) + 2 * 3.10"),
                AIMessage(content="Draft v2: Tile Museum, Gulbenkian and Tram 28, total 50.20 EUR (~562 SEK)."),
            ],
            "verifier": [
                structured("Review", issues=["The total leaves out the tram fares."], approved=False, summary="Total is incomplete."),
                structured("Review", issues=[], approved=True, summary="Plan matches the evidence."),
            ],
        },
        research_tools,
        planning_tools,
    )

    events = run_turn(graph, "t1", USER_REQUEST)

    routes = [e["next"] for e in events if e["type"] == "route"]
    assert routes == ["researcher", "planner", "verifier", "planner", "verifier", "finish"]

    tool_results = {e["tool"]: e["content"] for e in events if e["type"] == "tool_result"}
    assert "rain" in tool_results["get_weather_forecast"]
    assert "National Tile Museum" in tool_results["find_attractions"]
    assert "1,000.00 SEK = 89.30 EUR" in tool_results["convert_currency"]
    assert tool_results["calculator"].endswith("= 50.2")

    reviews = [e["approved"] for e in events if e["type"] == "review"]
    assert reviews == [False, True]
    assert events[-1] == {"type": "final", "content": "Draft v2: Tile Museum, Gulbenkian and Tram 28, total 50.20 EUR (~562 SEK)."}

    # The revising planner saw the verifier's issue; the verifier saw raw tool evidence.
    planner_prompts = [msgs[-1].text for role, msgs in llm.calls if role == "planner" and isinstance(msgs[-1], HumanMessage)]
    assert "The total leaves out the tram fares." in planner_prompts[-1]
    verifier_prompt = next(msgs[-1].text for role, msgs in llm.calls if role == "verifier")
    assert "convert_currency" in verifier_prompt and "89.30 EUR" in verifier_prompt

    # Only the user message and the final answer are persisted.
    history = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [type(m) for m in history] == [HumanMessage, AIMessage]


def test_follow_up_turn_sees_history_and_starts_with_a_clean_work_log(research_tools, planning_tools):
    llm, graph = make_graph(
        {
            "supervisor": [
                route("researcher", "How do I get from Lisbon airport to the centre?"),
                route("finish"),
                route("finish", reply="You're welcome, have a great trip!"),
            ],
            "researcher": [
                call_tool("search_travel_guides", "s1", query="airport to city centre", city="Lisbon"),
                AIMessage(content="Take the red metro line (about €1.85) (source: lisbon.md › Getting around)."),
            ],
        },
        research_tools,
        planning_tools,
    )

    first = run_turn(graph, "t2", "How do I get from the airport into Lisbon?")
    rag_result = next(e["content"] for e in first if e["type"] == "tool_result")
    assert "lisbon.md › Getting around" in rag_result
    assert first[-1]["content"].startswith("Take the red metro line")

    second = run_turn(graph, "t2", "Thanks!")
    assert second[-1] == {"type": "final", "content": "You're welcome, have a great trip!"}

    supervisor_prompt = [msgs[-1].text for role, msgs in llm.calls if role == "supervisor"][-1]
    assert "Assistant: Take the red metro line" in supervisor_prompt  # history carried over
    assert "## Work log for this message\n(nothing yet)" in supervisor_prompt  # per-turn state reset
    history = graph.get_state({"configurable": {"thread_id": "t2"}}).values["messages"]
    assert len(history) == 4


def test_unresolved_review_is_disclosed_in_the_answer(research_tools, planning_tools):
    _, graph = make_graph(
        {
            "supervisor": [route("planner", "Convert 1000 SEK to EUR."), route("verifier", "Check."), route("finish")],
            "planner": [AIMessage(content="1000 SEK is about 95 EUR.")],
            "verifier": [structured("Review", issues=["No conversion tool was used; the rate is invented."], approved=False, summary="Unverified rate.")],
        },
        research_tools,
        planning_tools,
    )
    final = run_turn(graph, "t3", "How much is 1000 SEK in euros?")[-1]["content"]
    assert final.startswith("1000 SEK is about 95 EUR.")
    assert "Heads-up from the verifier" in final and "the rate is invented" in final


def test_turn_is_cut_off_at_the_agent_run_limit(research_tools, planning_tools):
    _, graph = make_graph(
        {
            "supervisor": [route("researcher", "Look again.")] * 3,
            "researcher": [AIMessage(content=f"Attempt {i}") for i in range(2)],
        },
        research_tools,
        planning_tools,
        max_agent_runs=2,
    )
    events = run_turn(graph, "t4", "Tell me about Rome.")
    assert [e["next"] for e in events if e["type"] == "route"] == ["researcher", "researcher", "finish"]
    assert events[-1]["content"] == "Attempt 1"


def test_agent_step_limit_ends_the_run_cleanly(research_tools, planning_tools):
    # The researcher keeps calling tools. Steps: model, tools, model -> the limit of 3 stops
    # the run after the second model call, before its tool call can run.
    _, graph = make_graph(
        {
            "supervisor": [route("researcher", "Find indoor sights."), route("finish")],
            "researcher": [call_tool("find_attractions", f"a{i}", city="Rome", indoor=True) for i in range(5)],
        },
        research_tools,
        planning_tools,
        agent_step_limit=3,
    )
    events = run_turn(graph, "t5", "Indoor sights in Rome?")
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == len(results)  # every call the UI saw also gets a result
    assert results[-1]["content"] == "Not run: step limit reached."
    assert events[-1]["content"] == "I ran out of steps before finishing this task."


def _item(agent: str, output: str = "draft", approved: bool | None = None) -> WorkItem:
    return {"agent": agent, "task": "", "output": output, "tool_calls": [], "approved": approved, "issues": []}


@pytest.mark.parametrize(
    ("decision", "work_log", "expected_next"),
    [
        ("researcher", [], "researcher"),
        ("verifier", [], "finish"),  # nothing to verify
        ("verifier", [_item("planner")], "verifier"),
        ("verifier", [_item("planner"), _item("verifier", approved=False), _item("planner")], "verifier"),
        ("verifier", [_item("planner"), _item("verifier", approved=False)] * 2, "finish"),  # review limit
        ("planner", [_item("researcher")] * 6, "finish"),  # run limit
        ("finish", [_item("researcher")] * 9, "finish"),
    ],
)
def test_guardrails(decision, work_log, expected_next):
    proposed = SupervisorDecision(reason="r", next=decision, task="do it", reply="")
    assert apply_guardrails(proposed, work_log, max_agent_runs=6, max_reviews=2).next == expected_next


def test_guardrails_fill_in_a_missing_task():
    proposed = SupervisorDecision(reason="r", next="researcher", task="  ", reply="")
    assert apply_guardrails(proposed, [], max_agent_runs=6, max_reviews=2).task == "Answer the user's latest message."
