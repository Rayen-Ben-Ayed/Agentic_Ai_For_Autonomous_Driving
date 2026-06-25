import json

from simulation import telemetry


def test_derive_path():
    assert telemetry._derive_path("runs/foo.txt").endswith("foo.telemetry.jsonl")


def test_should_sample_first_last_and_every_n():
    # First and last ticks are always sampled.
    assert telemetry.should_sample(0, 80, lane_changed=False)
    assert telemetry.should_sample(79, 80, lane_changed=False)
    # A lane-id change is always sampled.
    assert telemetry.should_sample(5, 80, lane_changed=True)
    # Every Nth tick is sampled; off-beats are not.
    assert telemetry.should_sample(telemetry.TELEMETRY_LOG_EVERY, 80, lane_changed=False)
    assert not telemetry.should_sample(3, 80, lane_changed=False)


def test_write_row_noop_when_not_initialized():
    telemetry.close()
    telemetry.write_row({"a": 1})  # must not raise
    assert not telemetry.is_active()


def test_init_writes_rows(tmp_path):
    log_path = tmp_path / "run.txt"
    path = telemetry.init(str(log_path))
    try:
        assert path is not None and path.endswith(".telemetry.jsonl")
        assert telemetry.is_active()
        telemetry.write_row({"step": 1, "tick": 0})
        telemetry.write_row({"step": 1, "tick": 1})
    finally:
        telemetry.close()
    assert not telemetry.is_active()

    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln]
    assert len(lines) == 2
    assert json.loads(lines[0])["tick"] == 0
    assert json.loads(lines[1])["tick"] == 1
