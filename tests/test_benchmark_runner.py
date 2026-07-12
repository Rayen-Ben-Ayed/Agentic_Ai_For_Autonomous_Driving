from evaluation.benchmark_collector import RunBenchmarkResult
from evaluation.benchmark_runner import compute_determinism


def _run(action_sequence: list[str], success: bool = True) -> RunBenchmarkResult:
    return RunBenchmarkResult(
        scenario="1",
        run_index=1,
        success=success,
        collision_events=0,
        total_contact_substeps=0,
        first_collision_with=None,
        rule_violations=0,
        action_sequence=action_sequence,
    )


def test_determinism_identical_sequences():
    runs = [
        _run(["stop", "follow_lane", "follow_lane"]),
        _run(["stop", "follow_lane", "follow_lane"]),
    ]
    det = compute_determinism(runs)
    assert det["fully_identical_sequences"] is True
    assert det["unique_sequences"] == 1
    assert det["per_step_agreement"]["1"]["agreement_rate"] == 1.0


def test_determinism_mixed_sequences():
    runs = [
        _run(["stop", "follow_lane"]),
        _run(["yield", "follow_lane"]),
        _run(["stop", "follow_lane"]),
    ]
    det = compute_determinism(runs)
    assert det["fully_identical_sequences"] is False
    assert det["unique_sequences"] == 2
    assert det["per_step_agreement"]["1"]["mode_action"] == "stop"
    assert det["per_step_agreement"]["1"]["agreement_rate"] == round(2 / 3, 3)


def test_determinism_ignores_failed_runs():
    runs = [
        _run(["stop"], success=False),
        _run(["yield"], success=True),
    ]
    det = compute_determinism(runs)
    assert det["comparable_runs"] == 1
