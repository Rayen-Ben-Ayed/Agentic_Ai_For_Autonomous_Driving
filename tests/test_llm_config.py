import importlib

import pytest


@pytest.fixture()
def clean_llm_env(monkeypatch):
    # Prevent reload from re-reading the developer's local .env file.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    for key in (
        "LLM_MODEL",
        "LLM_PROVIDER",
        "GROQ_MODEL",
        "CEREBRAS_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_REMOTE_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def _reload():
    import agent.llm_config as lc

    importlib.reload(lc)
    return lc


def test_provider_model_from_env(clean_llm_env, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    lc = _reload()
    assert lc.resolve_model("groq") == "llama-3.1-8b-instant"


def test_global_llm_model_overrides_provider(clean_llm_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("GROQ_MODEL", "ignored")
    lc = _reload()
    assert lc.resolve_model("groq") == "custom-model"
    assert lc.resolve_model("cerebras") == "custom-model"


def test_cerebras_and_ollama_defaults(clean_llm_env):
    lc = _reload()
    assert lc.resolve_model("cerebras") == "llama3.1-8b"
    assert lc.resolve_model("ollama") == "llama3"
    assert lc.resolve_model("ollama-remote") == "llama3.1:8b"
