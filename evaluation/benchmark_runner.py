"""Aggregate benchmark runs and compute cross-run determinism metrics."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evaluation.benchmark_collector import DecisionRecord, RunBenchmarkResult


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def compute_determinism(runs: list[RunBenchmarkResult]) -> dict[str, Any]:
    """Compare chosen actions across runs for the same scenario."""
    sequences = [run.action_sequence for run in runs if run.success]
    if not sequences:
        return {
            "comparable_runs": 0,
            "fully_identical_sequences": False,
            "unique_sequences": 0,
            "per_step_agreement": {},
            "sequences": [run.action_sequence for run in runs],
        }

    max_len = max(len(seq) for seq in sequences)
    per_step: dict[str, Any] = {}
    for step_idx in range(max_len):
        step_num = step_idx + 1
        actions = [
            seq[step_idx] if step_idx < len(seq) else None for seq in sequences
        ]
        counts = Counter(actions)
        mode_action, mode_count = counts.most_common(1)[0]
        per_step[str(step_num)] = {
            "mode_action": mode_action,
            "agreement_rate": round(mode_count / len(actions), 3),
            "action_counts": dict(counts),
        }

    unique = {tuple(seq) for seq in sequences}
    return {
        "comparable_runs": len(sequences),
        "fully_identical_sequences": len(unique) == 1,
        "unique_sequences": len(unique),
        "per_step_agreement": per_step,
        "sequences": [run.action_sequence for run in runs],
    }


def _decision_metric(runs: list[RunBenchmarkResult], attr: str) -> dict[str, float]:
    values = [
        float(getattr(d, attr))
        for run in runs
        for d in run.decisions
    ]
    return {
        "mean": _mean(values),
        "min": round(min(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
        "p95": _percentile(values, 95),
    }


def aggregate_runs(
    scenario: str,
    repeats: int,
    runs: list[RunBenchmarkResult],
    *,
    llm_provider: str,
    llm_model: Optional[str],
) -> dict[str, Any]:
    successful = [r for r in runs if r.success]
    failed = [r for r in runs if not r.success]

    all_decisions: list[DecisionRecord] = [d for run in runs for d in run.decisions]

    return {
        "scenario": scenario,
        "repeats": repeats,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "completed_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "safety": {
            "collision_free_rate": round(
                sum(1 for r in successful if r.collision_events == 0)
                / len(successful),
                3,
            )
            if successful
            else 0.0,
            "avg_collision_events": _mean(
                [float(r.collision_events) for r in successful]
            ),
            "avg_contact_substeps": _mean(
                [float(r.total_contact_substeps) for r in successful]
            ),
        },
        "decision_time_ms": _decision_metric(runs, "decision_time_ms"),
        "llm_response_time_ms": _decision_metric(runs, "llm_response_time_ms"),
        "mcp_tool_calls_per_decision": _decision_metric(runs, "mcp_tool_calls"),
        "llm_calls_per_decision": _decision_metric(runs, "llm_calls"),
        "token_usage": {
            "total_prompt_tokens": sum(d.llm_prompt_tokens for d in all_decisions),
            "total_completion_tokens": sum(
                d.llm_completion_tokens for d in all_decisions
            ),
            "total_tokens": sum(d.llm_total_tokens for d in all_decisions),
            "avg_tokens_per_decision": _mean(
                [float(d.llm_total_tokens) for d in all_decisions]
            ),
            "note": (
                "Token counts are reported when the provider returns usage metadata "
                "(Groq/Cerebras/OpenAI-compatible). Ollama/Gemini may report 0."
            ),
        },
        "actions": {
            "total_acceptances": sum(d.action_acceptances for d in all_decisions),
            "total_rejections": sum(d.action_rejections for d in all_decisions),
            "acceptance_rate": round(
                sum(d.action_acceptances for d in all_decisions)
                / max(
                    1,
                    sum(d.action_acceptances + d.action_rejections for d in all_decisions),
                ),
                3,
            ),
        },
        "determinism": compute_determinism(runs),
        "runs": [run.to_dict() for run in runs],
        "errors": [run.error for run in failed if run.error],
    }


def save_benchmark_report(report: dict[str, Any], output_path: Optional[str] = None) -> str:
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent / "benchmark_results"
        out_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = report.get("scenario", "unknown")
        output_path = str(out_dir / f"benchmark_s{scenario}_{stamp}.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return str(path)


def print_benchmark_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"Benchmark — scenario {report['scenario']} x {report['repeats']} runs")
    print("=" * 60)
    print(
        f"Runs: {report['successful_runs']}/{report['completed_runs']} successful | "
        f"LLM: {report['llm_provider']} ({report.get('llm_model') or 'n/a'})"
    )
    safety = report["safety"]
    print(
        f"Collisions: collision-free rate {safety['collision_free_rate']:.0%} | "
        f"avg events {safety['avg_collision_events']}"
    )
    dt = report["decision_time_ms"]
    llm = report["llm_response_time_ms"]
    mcp = report["mcp_tool_calls_per_decision"]
    print(
        f"Decision time (ms): mean={dt['mean']} p95={dt['p95']} | "
        f"LLM response mean={llm['mean']} p95={llm['p95']}"
    )
    print(
        f"MCP tool calls/decision: mean={mcp['mean']} | "
        f"LLM calls/decision: mean={report['llm_calls_per_decision']['mean']}"
    )
    tokens = report["token_usage"]
    print(
        f"Tokens: total={tokens['total_tokens']} "
        f"(prompt={tokens['total_prompt_tokens']}, "
        f"completion={tokens['total_completion_tokens']}) | "
        f"avg/decision={tokens['avg_tokens_per_decision']}"
    )
    actions = report["actions"]
    print(
        f"Actions: accepted={actions['total_acceptances']} "
        f"rejected={actions['total_rejections']} "
        f"(accept rate {actions['acceptance_rate']:.0%})"
    )
    det = report["determinism"]
    print(
        f"Determinism: identical sequences={det['fully_identical_sequences']} | "
        f"unique sequences={det['unique_sequences']}/{det['comparable_runs']}"
    )
    if det.get("per_step_agreement"):
        print("Per-step action agreement:")
        for step, info in sorted(
            det["per_step_agreement"].items(), key=lambda kv: int(kv[0])
        ):
            print(
                f"  step {step}: mode={info['mode_action']} "
                f"agreement={info['agreement_rate']:.0%} counts={info['action_counts']}"
            )
    print("=" * 60 + "\n")
