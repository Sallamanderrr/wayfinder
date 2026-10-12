"""Multi-agent orchestration with LangGraph: a supervisor that routes between specialists.

    START -> supervisor --(LLM decides)--> researcher | planner | verifier
                 ^                                    |
                 +------------ back to supervisor ----+
             ...until the supervisor picks "finish" -> finalize -> END

Agents never call each other directly. They share a per-turn work log in the graph state
(a "blackboard"): every agent run appends a WorkItem, the supervisor reads the log to
decide what happens next, and the next agent receives it as context. Only the final
answer is added to the persistent `messages` history, which the checkpointer stores per
conversation thread.

The LLM makes the routing decisions; code enforces hard limits (max agent runs, max
reviews) so that every turn terminates, whatever the model decides.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from . import prompts

WORKERS = ("researcher", "planner")
FALLBACK_ANSWER = (
    "Sorry, I couldn't put together an answer to that. Could you rephrase it or add a bit more detail?"
)


# --- State ------------------------------------------------------------------------------


class ToolCallRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    result: str


class WorkItem(TypedDict):
    agent: str  # researcher | planner | verifier
    task: str  # the supervisor's instruction
    output: str  # the agent's answer; for the verifier, its one-line summary
    tool_calls: list[ToolCallRecord]
    approved: bool | None  # verifier only
    issues: list[str]  # verifier only


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # chat history, kept across turns
    work_log: list[WorkItem]  # reset at the start of every turn
    next: str  # the supervisor's latest routing decision
    task: str  # the instruction for the agent that runs next
    reply: str  # the supervisor's own reply, for turns that need no agent


def turn_input(message: str) -> dict[str, Any]:
    """Graph input for one user turn: appends to the history and resets per-turn fields."""
    return {"messages": [HumanMessage(content=message)], "work_log": [], "next": "", "task": "", "reply": ""}


# --- Structured outputs ------------------------------------------------------------------


# Field order matters: models generate JSON fields in order, so the reasoning field comes
# before the decision it justifies.


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""

    reason: str = Field(description="One short sentence explaining the decision. Shown to the user.")
    next: Literal["researcher", "planner", "verifier", "finish"] = Field(
        description="Who acts next, or 'finish' to answer the user now."
    )
    task: str = Field(
        description="A self-contained instruction for that agent, with city, dates (YYYY-MM-DD), "
        "travellers, budget, currency and preferences. Empty string when finishing."
    )
    reply: str = Field(
        description="Only when finishing before any agent has drafted an answer in this turn "
        "(greetings, clarifying questions): the full reply to the user. Otherwise an empty string."
    )


class Review(BaseModel):
    """The verifier's verdict on the latest draft."""

    issues: list[str] = Field(description="Concrete, fixable problems. Empty when there are none.")
    approved: bool = Field(
        description="True if the draft is accurate, grounded in the evidence and meets the user's constraints."
    )
    summary: str = Field(description="One sentence summarising the review. Shown to the user.")


# --- Graph -------------------------------------------------------------------------------


def build_graph(
    llm: BaseChatModel,
    research_tools: Sequence[BaseTool],
    planning_tools: Sequence[BaseTool],
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    today: Callable[[], date] = date.today,
    max_agent_runs: int = 6,
    max_reviews: int = 2,
    agent_step_limit: int = 12,
):
    # Workers are ReAct-style tool-calling agents. They are stateless (checkpointer=False):
    # every run starts fresh from the context we pass in; the conversation lives in the outer graph.
    researcher = create_agent(
        llm, list(research_tools), system_prompt=prompts.RESEARCHER, name="researcher", checkpointer=False
    )
    planner = create_agent(
        llm, list(planning_tools), system_prompt=prompts.PLANNER, name="planner", checkpointer=False
    )

    graph = StateGraph(TravelState)
    graph.add_node("supervisor", _supervisor_node(llm, today, max_agent_runs, max_reviews))
    graph.add_node("researcher", _worker_node("researcher", researcher, today, agent_step_limit))
    graph.add_node("planner", _worker_node("planner", planner, today, agent_step_limit))
    graph.add_node("verifier", _verifier_node(llm, today))
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"researcher": "researcher", "planner": "planner", "verifier": "verifier", "finish": "finalize"},
    )
    for agent in ("researcher", "planner", "verifier"):
        graph.add_edge(agent, "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# --- Nodes -------------------------------------------------------------------------------


def _supervisor_node(llm: BaseChatModel, today: Callable[[], date], max_agent_runs: int, max_reviews: int):
    router = llm.with_structured_output(SupervisorDecision)

    def supervisor(state: TravelState) -> dict[str, Any]:
        context = _supervisor_context(state, today(), max_agent_runs, max_reviews)
        decision = router.invoke([SystemMessage(content=prompts.SUPERVISOR), HumanMessage(content=context)])
        decision = apply_guardrails(decision, state["work_log"], max_agent_runs, max_reviews)
        get_stream_writer()(
            {"type": "route", "next": decision.next, "task": decision.task, "reason": decision.reason}
        )
        return {"next": decision.next, "task": decision.task, "reply": decision.reply}

    return supervisor


def apply_guardrails(
    decision: SupervisorDecision, work_log: list[WorkItem], max_agent_runs: int, max_reviews: int
) -> SupervisorDecision:
    """Deterministic limits on top of the LLM's routing, so every turn terminates."""
    if decision.next == "finish":
        return decision
    if len(work_log) >= max_agent_runs:
        return _finish(decision, f"Stopping: reached the limit of {max_agent_runs} agent runs for one message.")
    if decision.next == "verifier":
        if not latest_draft(work_log)[1]:
            return _finish(decision, "Stopping: there is no draft to verify.")
        reviews = sum(1 for item in work_log if item["agent"] == "verifier")
        if reviews >= max_reviews:
            return _finish(decision, f"Stopping: the draft has already been reviewed {reviews} times.")
    if not decision.task.strip():
        return decision.model_copy(update={"task": "Answer the user's latest message."})
    return decision


def _finish(decision: SupervisorDecision, reason: str) -> SupervisorDecision:
    return decision.model_copy(update={"next": "finish", "task": "", "reason": reason})


def _worker_node(name: str, agent: Any, today: Callable[[], date], step_limit: int):
    def worker(state: TravelState) -> dict[str, Any]:
        writer = get_stream_writer()
        context = _worker_context(state, today())
        output, tool_calls = _run_agent(agent, name, context, writer, step_limit)
        writer({"type": "agent_output", "agent": name, "content": output})
        item: WorkItem = {
            "agent": name,
            "task": state["task"],
            "output": output,
            "tool_calls": tool_calls,
            "approved": None,
            "issues": [],
        }
        return {"work_log": [*state["work_log"], item]}

    return worker


def _run_agent(
    agent: Any, name: str, context: str, writer: Callable[[Any], None], step_limit: int
) -> tuple[str, list[ToolCallRecord]]:
    """Run one tool-calling agent, forwarding each tool call and result as a UI event."""
    pending: dict[str, ToolCallRecord] = {}
    records: list[ToolCallRecord] = []
    output = ""
    try:
        for update in agent.stream(
            {"messages": [HumanMessage(content=context)]},
            {"recursion_limit": step_limit},
            stream_mode="updates",
        ):
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for message in node_update.get("messages", []):
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            pending[call.get("id") or call["name"]] = {
                                "tool": call["name"],
                                "args": call["args"],
                                "result": "",
                            }
                            writer({"type": "tool_call", "agent": name, "tool": call["name"], "args": call["args"]})
                        if not message.tool_calls:
                            output = message.text
                    elif isinstance(message, ToolMessage):
                        fallback: ToolCallRecord = {"tool": message.name or "tool", "args": {}, "result": ""}
                        record = pending.pop(message.tool_call_id, fallback)
                        record["result"] = message.text
                        records.append(record)
                        writer(
                            {
                                "type": "tool_result",
                                "agent": name,
                                "tool": record["tool"],
                                "content": _truncate(message.text, 600),
                            }
                        )
    except GraphRecursionError:
        output = output or "I ran out of steps before finishing this task."
        for record in pending.values():  # requested, but cut off before they could run
            writer({"type": "tool_result", "agent": name, "tool": record["tool"], "content": "Not run: step limit reached."})
    return output, records


def _verifier_node(llm: BaseChatModel, today: Callable[[], date]):
    reviewer = llm.with_structured_output(Review)

    def verifier(state: TravelState) -> dict[str, Any]:
        context = _verifier_context(state, today())
        review = reviewer.invoke([SystemMessage(content=prompts.VERIFIER), HumanMessage(content=context)])
        get_stream_writer()(
            {"type": "review", "approved": review.approved, "issues": review.issues, "summary": review.summary}
        )
        item: WorkItem = {
            "agent": "verifier",
            "task": state["task"],
            "output": review.summary,
            "tool_calls": [],
            "approved": review.approved,
            "issues": review.issues,
        }
        return {"work_log": [*state["work_log"], item]}

    return verifier


def _finalize_node(state: TravelState) -> dict[str, Any]:
    work_log = state["work_log"]
    draft_index, draft = latest_draft(work_log)
    if draft:
        answer = draft
        review_index, review = _latest_review(work_log)
        if review is not None and review_index > draft_index and not review["approved"]:
            # Out of revision budget: show the answer, but don't hide the verifier's concerns.
            answer += "\n\n> **Heads-up from the verifier:** this answer may contain problems:\n" + "\n".join(
                f"> - {issue}" for issue in review["issues"]
            )
    else:
        answer = state["reply"].strip() or FALLBACK_ANSWER
    get_stream_writer()({"type": "final", "content": answer})
    return {"messages": [AIMessage(content=answer)]}


# --- Context rendering -------------------------------------------------------------------


def latest_draft(work_log: list[WorkItem]) -> tuple[int, str]:
    """The most recent researcher/planner answer, with its index in the work log."""
    for index in range(len(work_log) - 1, -1, -1):
        item = work_log[index]
        if item["agent"] in WORKERS and item["output"].strip():
            return index, item["output"]
    return -1, ""


def _latest_review(work_log: list[WorkItem]) -> tuple[int, WorkItem | None]:
    for index in range(len(work_log) - 1, -1, -1):
        if work_log[index]["agent"] == "verifier":
            return index, work_log[index]
    return -1, None


def _supervisor_context(state: TravelState, today: date, max_agent_runs: int, max_reviews: int) -> str:
    reviews = sum(1 for item in state["work_log"] if item["agent"] == "verifier")
    return "\n\n".join(
        [
            _header(state, today),
            f"## Work log for this message\n{_format_work_log(state['work_log'], output_limit=1500)}",
            f"## Limits\nAgent runs used: {len(state['work_log'])} of {max_agent_runs}. "
            f"Reviews used: {reviews} of {max_reviews}.",
        ]
    )


def _worker_context(state: TravelState, today: date) -> str:
    return "\n\n".join(
        [
            _header(state, today),
            f"## Work already done for this message\n{_format_work_log(state['work_log'], output_limit=None)}",
            f"## Your task (from the supervisor)\n{state['task']}",
        ]
    )


def _verifier_context(state: TravelState, today: date) -> str:
    work_log = state["work_log"]
    draft_index, draft = latest_draft(work_log)
    author = work_log[draft_index]["agent"] if draft_index >= 0 else "team"
    notes = [item["output"] for item in work_log[:draft_index] if item["agent"] == "researcher"]
    return "\n\n".join(
        [
            _header(state, today),
            f"## Evidence: tool results gathered for this message\n{_format_evidence(work_log)}",
            "## Research notes\n" + ("\n\n---\n\n".join(notes) if notes else "(none besides the draft)"),
            f"## Draft to review (written by the {author})\n{draft}",
        ]
    )


def _header(state: TravelState, today: date) -> str:
    messages = state["messages"]
    latest = messages[-1].text if messages else ""
    return (
        f"Today is {today.strftime('%A %Y-%m-%d')}.\n\n"
        f"## Conversation so far\n{_format_history(messages[:-1])}\n\n"
        f"## The user's latest message\n{latest}"
    )


def _format_history(messages: list[AnyMessage], max_messages: int = 6, max_chars: int = 3000) -> str:
    recent = messages[-max_messages:]
    if not recent:
        return "(none: this is the first message)"
    return "\n\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_truncate(m.text, max_chars)}" for m in recent
    )


def _format_work_log(work_log: list[WorkItem], output_limit: int | None) -> str:
    if not work_log:
        return "(nothing yet)"
    blocks = []
    for number, item in enumerate(work_log, start=1):
        if item["agent"] == "verifier":
            verdict = "APPROVED" if item["approved"] else "CHANGES REQUESTED"
            issues = "".join(f"\n- {issue}" for issue in item["issues"])
            blocks.append(f"### {number}. verifier: {verdict}\n{item['output']}{issues}")
        else:
            output = item["output"] if output_limit is None else _truncate(item["output"], output_limit)
            tools_used = ", ".join(call["tool"] for call in item["tool_calls"]) or "none"
            blocks.append(
                f"### {number}. {item['agent']} (tools used: {tools_used})\n"
                f"Task: {item['task']}\nResult:\n{output}"
            )
    return "\n\n".join(blocks)


def _format_evidence(work_log: list[WorkItem]) -> str:
    lines = []
    for item in work_log:
        for call in item["tool_calls"]:
            args = json.dumps(call["args"], ensure_ascii=False)
            lines.append(f"- [{item['agent']}] {call['tool']}({args}) returned:\n{_truncate(call['result'], 2500)}")
    return "\n".join(lines) or "(no tool results)"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
