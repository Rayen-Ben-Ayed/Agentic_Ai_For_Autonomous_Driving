"""Collect per-run benchmark metrics for agent decisions, LLM calls, and MCP tools."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class LLMCallRecord:
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class DecisionRecord:
    step: int
    action: Optional[str]
    decision_time_ms: float
    llm_calls: int
    llm_response_time_ms: float
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int
    mcp_tool_calls: int
    action_acceptances: int
    action_rejections: int
    attempted_actions: list[str] = field(default_factory=list)


@dataclass
class RunBenchmarkResult:
    scenario: str
    run_index: int
    success: bool
    collision_events: int
    total_contact_substeps: int
    first_collision_with: Optional[str]
    rule_violations: int
    decisions: list[DecisionRecord] = field(default_factory=list)
    action_sequence: list[str] = field(default_factory=list)
    error: Optional[str] = None
    log_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decisions"] = [asdict(d) for d in self.decisions]
        return payload


class BenchmarkCollector:
    """Process-global collector for one simulation run."""

    def __init__(self) -> None:
        self._active = False
        self._current_step = 0
        self._decision_start: Optional[float] = None
        self._decision_llm_calls: list[LLMCallRecord] = []
        self._decision_mcp_tool_calls = 0
        self._decision_acceptances = 0
        self._decision_rejections = 0
        self._decision_attempted_actions: list[str] = []
        self.decisions: list[DecisionRecord] = []
        self.action_sequence: list[str] = []

    def activate(self) -> None:
        self._active = True
        self.decisions.clear()
        self.action_sequence.clear()

    @property
    def active(self) -> bool:
        return self._active

    def begin_decision(self, step: int) -> None:
        if not self._active:
            return
        self._current_step = step
        self._decision_start = time.perf_counter()
        self._decision_llm_calls = []
        self._decision_mcp_tool_calls = 0
        self._decision_acceptances = 0
        self._decision_rejections = 0
        self._decision_attempted_actions = []

    def record_llm_call(
        self,
        latency_ms: float,
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        if not self._active:
            return
        self._decision_llm_calls.append(
            LLMCallRecord(
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )

    def record_mcp_tool_call(self, tool_name: str, arguments: Optional[dict] = None) -> None:
        if not self._active:
            return
        self._decision_mcp_tool_calls += 1
        if tool_name == "execute_action" and arguments:
            action = arguments.get("action")
            if isinstance(action, str):
                self._decision_attempted_actions.append(action)

    def record_action_acceptance(self, action: str) -> None:
        if not self._active:
            return
        self._decision_acceptances += 1

    def record_action_rejection(self, action: str) -> None:
        if not self._active:
            return
        self._decision_rejections += 1

    def end_decision(self, action: Optional[str]) -> None:
        if not self._active or self._decision_start is None:
            return
        decision_time_ms = (time.perf_counter() - self._decision_start) * 1000
        llm_response_time_ms = sum(c.latency_ms for c in self._decision_llm_calls)
        prompt_tokens = sum(c.prompt_tokens or 0 for c in self._decision_llm_calls)
        completion_tokens = sum(c.completion_tokens or 0 for c in self._decision_llm_calls)
        total_tokens = sum(c.total_tokens or 0 for c in self._decision_llm_calls)

        record = DecisionRecord(
            step=self._current_step,
            action=action,
            decision_time_ms=round(decision_time_ms, 2),
            llm_calls=len(self._decision_llm_calls),
            llm_response_time_ms=round(llm_response_time_ms, 2),
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            llm_total_tokens=total_tokens,
            mcp_tool_calls=self._decision_mcp_tool_calls,
            action_acceptances=self._decision_acceptances,
            action_rejections=self._decision_rejections,
            attempted_actions=list(self._decision_attempted_actions),
        )
        self.decisions.append(record)
        if action:
            self.action_sequence.append(action)

        self._decision_start = None
