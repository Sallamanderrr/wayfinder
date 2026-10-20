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


SIGHTS = [
    {"name": "National Tile Museum", "price_per_person": 8},
    {"name": "Calouste Gulbenkian Museum", "price_per_person": 14},
    {"name": "Tram 28", "price_per_person": 3.1},
]


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
                route("planner", "Fix the issue: the budget leaves out the tram fares."),
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
                call_tool("budget_breakdown", "b1", items=SIGHTS[:2], travellers=2, currency_code="EUR", budget=1000, budget_currency="SEK"),
                AIMessage(content="Draft v1: Tile Museum and Gulbenkian, 492.57 SEK of your 1000 SEK."),
                call_tool("budget_breakdown", "b2", items=SIGHTS, travellers=2, currency_code="EUR", budget=1000, budget_currency="SEK"),
                AIMessage(content="Draft v2: Tile Museum, Gulbenkian and Tram 28, 561.98 SEK. 438.02 SEK left."),
            ],
            "verifier": [
                structured("Review", issues=["The budget leaves out the tram fares."], approved=False, summary="Budget is incomplete."),
                structured("Review", issues=[], approved=True, summary="Plan matches the evidence."),
            ],
        },
        research_tools,
        planning_tools,
    )

    events = run_turn(graph, "t1", USER_REQUEST)

    routes = [e["next"] for e in events if e["type"] == "route"]
    assert routes == ["researcher", "planner", "verifier", "planner", "verifier", "finish"]

    # Each result carries its call's id, so the UI pairs parallel calls correctly.
    results_by_id = {e["id"]: e["content"] for e in events if e["type"] == "tool_result"}
    assert results_by_id["w1"].startswith("Daily forecast for Lisbon")
    assert "attraction(s) in Lisbon" in results_by_id["a1"]
    assert {e["id"] for e in events if e["type"] == "tool_call"} == set(results_by_id)

    tool_results = {e["tool"]: e["content"] for e in events if e["type"] == "tool_result"}
    assert "rain" in tool_results["get_weather_forecast"]
    assert "National Tile Museum" in tool_results["find_attractions"]
    assert "| **Total** | | **50.20** | **561.98** |" in tool_results["budget_breakdown"]

    reviews = [e["approved"] for e in events if e["type"] == "review"]
    assert reviews == [False, True]
    assert events[-1] == {"type": "final", "content": "Draft v2: Tile Museum, Gulbenkian and Tram 28, 561.98 SEK. 438.02 SEK left."}

    # The revising planner saw the verifier's issue; the verifier saw raw tool evidence.
    planner_prompts = [msgs[-1].text for role, msgs in llm.calls if role == "planner" and isinstance(msgs[-1], HumanMessage)]
    assert "The budget leaves out the tram fares." in planner_prompts[-1]
    verifier_prompt = next(msgs[-1].text for role, msgs in llm.calls if role == "verifier")
    assert "budget_breakdown" in verifier_prompt and "492.57" in verifier_prompt

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


def test_code_check_overrides_an_approval_when_figures_are_made_up(research_tools, planning_tools):
    _, graph = make_graph(
        {
            "supervisor": [route("planner", "Budget 2 x 8 EUR in SEK."), route("verifier", "Check."), route("finish")],
            "planner": [
                call_tool("convert_currency", "c1", amount=1, from_currency="EUR", to_currency="SEK"),
                AIMessage(content="Two tickets: 179.20 SEK."),  # 16 x 11.1948 = 179.12, computed by hand
            ],
            "verifier": [structured("Review", issues=[], approved=True, summary="Looks right.")],
        },
        research_tools,
        planning_tools,
    )
    events = run_turn(graph, "t6", "Two 8 EUR tickets in SEK?")
    review = next(e for e in events if e["type"] == "review")
    assert review["approved"] is False
    assert "179.20" in review["issues"][0]
    assert "Heads-up from the verifier" in events[-1]["content"]


def test_hand_converted_budget_is_rejected_and_fixed(research_tools, planning_tools):
    # Replays a live run: the planner converted 1,500 SEK to "136 EUR" by hand and passed
    # that to budget_breakdown. The code check rejects it; the revision passes.
    items = [{"name": "National Tile Museum", "price_per_person": 8}]
    _, graph = make_graph(
        {
            "supervisor": [
                route("researcher", "Lisbon indoor sights."),
                route("planner", "Plan within 1,500 SEK."),
                route("verifier", "Check."),
                route("planner", "Fix the budget inputs."),
                route("verifier", "Re-check."),
                route("finish"),
            ],
            "researcher": [
                call_tool("find_attractions", "a1", city="Lisbon", indoor=True, max_price=10),
                AIMessage(content="National Tile Museum, 8 EUR (attractions database)."),
            ],
            "planner": [
                call_tool("budget_breakdown", "b1", items=items, travellers=2, currency_code="EUR", budget=136, budget_currency="EUR"),
                AIMessage(content="Tile Museum: 16.00 EUR. Within budget: 120.00 EUR left of 136.00 EUR."),
                call_tool("budget_breakdown", "b2", items=items, travellers=2, currency_code="EUR", budget=1500, budget_currency="SEK"),
                AIMessage(content="Tile Museum: 179.12 SEK for two. Within budget: 1,320.88 SEK left of 1,500.00 SEK."),
            ],
            "verifier": [
                structured("Review", issues=[], approved=True, summary="Looks right."),
                structured("Review", issues=[], approved=True, summary="Looks right."),
            ],
        },
        research_tools,
        planning_tools,
    )
    events = run_turn(graph, "t7", "Plan a Lisbon day for 2. Our budget for sights is 1,500 SEK.")
    first, second = [e for e in events if e["type"] == "review"]
    assert first["approved"] is False and "budget 136" in first["issues"][0]
    assert second["approved"] is True
    assert "1,320.88 SEK left" in events[-1]["content"]  # 1,500 - 16 x 11.1948 = 1,320.88


def test_follow_up_may_reuse_prices_from_the_previous_answer(research_tools, planning_tools):
    # Replays a live run: the follow-up added a Monday, this turn's research only covered
    # Monday (the Tile Museum is closed then), and the planner reused the museum's price from
    # the previous answer. That price was checked last turn, so it must not be flagged now.
    tile = {"name": "National Tile Museum", "price_per_person": 8}
    gulbenkian = {"name": "Calouste Gulbenkian Museum", "price_per_person": 14}
    budget = {"travellers": 2, "currency_code": "EUR", "budget": 1500, "budget_currency": "SEK"}
    _, graph = make_graph(
        {
            "supervisor": [
                route("researcher", "Lisbon indoor sights for Saturday."),
                route("planner", "Plan Saturday within 1,500 SEK."),
                route("finish"),
                route("researcher", "Lisbon indoor sights for Monday."),
                route("planner", "Add Monday to the plan."),
                route("verifier", "Check."),
                route("finish"),
            ],
            "researcher": [
                call_tool("find_attractions", "a1", city="Lisbon", indoor=True, max_price=10, open_on="Saturday"),
                AIMessage(content="National Tile Museum, 8 EUR (attractions database)."),
                call_tool("find_attractions", "a2", city="Lisbon", indoor=True, open_on="Monday"),
                AIMessage(content="Monday: Calouste Gulbenkian Museum, 14 EUR (attractions database)."),
            ],
            "planner": [
                call_tool("budget_breakdown", "b1", items=[tile], **budget),
                AIMessage(content="Saturday: National Tile Museum, 8 EUR per person, 179.12 SEK for two."),
                call_tool("budget_breakdown", "b2", items=[tile, gulbenkian], **budget),
                AIMessage(content="Saturday Tile Museum, Monday Gulbenkian: 492.57 SEK, 1,007.43 SEK left."),
            ],
            "verifier": [structured("Review", issues=[], approved=True, summary="Looks right.")],
        },
        research_tools,
        planning_tools,
    )
    run_turn(graph, "t9", "Plan Saturday in Lisbon for 2. Our budget for sights is 1,500 SEK.")
    events = run_turn(graph, "t9", "Add a Monday.")
    monday_research = next(e["content"] for e in events if e["type"] == "tool_result")
    assert "Tile Museum" not in monday_research  # so its price only exists in the earlier answer
    review = next(e for e in events if e["type"] == "review")
    assert review["approved"] is True, review["issues"]
    assert "Heads-up" not in events[-1]["content"]


@pytest.mark.parametrize(
    ("revision", "warned"),
    [
        ("Revised: two tickets cost 179.12 SEK.", False),  # backed by the conversion: shown as is
        ("Revised: two tickets cost 180.00 SEK.", True),  # made up: flagged even without a review
    ],
)
def test_revision_after_the_last_allowed_review_is_still_checked(research_tools, planning_tools, revision, warned):
    # max_reviews=1: the planner revises after a failed review, and the supervisor may not
    # call the verifier again. The final step still runs the code checks on the revision.
    _, graph = make_graph(
        {
            "supervisor": [
                route("planner", "Two 8 EUR tickets in SEK."),
                route("verifier", "Check."),
                route("planner", "Fix the issues."),
                route("verifier", "Re-check."),  # blocked by the review limit -> finish
            ],
            "planner": [
                call_tool("convert_currency", "c1", amount=16, from_currency="EUR", to_currency="SEK"),
                AIMessage(content="Two tickets cost 175.00 SEK."),
                AIMessage(content=revision),
            ],
            "verifier": [structured("Review", issues=["Total looks wrong."], approved=False, summary="Wrong total.")],
        },
        research_tools,
        planning_tools,
        max_reviews=1,
    )
    events = run_turn(graph, "t8", "Two 8 EUR tickets in SEK?")
    assert events[-2] == {"type": "route", "next": "finish", "task": "", "reason": "Stopping: the review limit (1) has been reached."}
    final = events[-1]["content"]
    assert final.startswith(revision)
    assert ("Heads-up from the verifier" in final) is warned


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
