import os

import pytest


@pytest.fixture()
def clean_step_env(monkeypatch):
    for key in ("NUM_STEPS", "STEP_INTERVAL_S", "MAX_LLM_TOOL_ROUNDS"):
        monkeypatch.delenv(key, raising=False)


def test_num_steps_from_env(clean_step_env, monkeypatch):
    monkeypatch.setenv("NUM_STEPS", "15")
    import simulation.timing_config as tc

    import importlib

    importlib.reload(tc)
    assert tc.NUM_STEPS == 15
    assert tc.simulated_duration_s() == 15 * tc.STEP_INTERVAL_S


def test_num_steps_minimum_one(clean_step_env, monkeypatch):
    monkeypatch.setenv("NUM_STEPS", "0")
    import simulation.timing_config as tc

    import importlib

    importlib.reload(tc)
    assert tc.NUM_STEPS == 1
