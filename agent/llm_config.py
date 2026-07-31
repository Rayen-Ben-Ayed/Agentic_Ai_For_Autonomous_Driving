"""LLM provider and model configuration (read from environment)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

SUPPORTED_PROVIDERS = frozenset(
    {"groq", "cerebras", "ollama", "ollama-remote", "academic_cloud"}
)

# Per-provider model env var and default when unset.
PROVIDER_MODEL_ENV: dict[str, tuple[str, str]] = {
    "groq": ("GROQ_MODEL", "openai/gpt-oss-120b"),
    "cerebras": ("CEREBRAS_MODEL", "llama3.1-8b"),
    "ollama": ("OLLAMA_MODEL", "llama3"),
    "ollama-remote": ("OLLAMA_REMOTE_MODEL", "llama3.1:8b"),
    "academic_cloud": ("ACADEMIC_CLOUD_MODEL", "OpenAI GPT OSS 120B"),
}

PROVIDER_BASE_URL_ENV: dict[str, tuple[str, str]] = {
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "ollama-remote": (
        "OLLAMA_REMOTE_BASE_URL",
        "http://localhost:11434/v1",
    ),
    "academic_cloud": ("ACADEMIC_CLOUD_BASE_URL", "https://chat.kiconnect.nrw/api/v1"),
}


def _strip(value: str | None) -> str:
    return (value or "").strip()


def resolve_provider(explicit: str | None = None) -> str:
    provider = _strip(explicit or os.getenv("LLM_PROVIDER", "groq")).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    return provider


def resolve_model(provider: str) -> str:
    """Return the model id for `provider`.

    Priority: LLM_MODEL (global) > provider-specific env > built-in default.
    """
    provider = provider.lower()
    global_model = _strip(os.getenv("LLM_MODEL"))
    if global_model:
        return global_model

    env_key, default = PROVIDER_MODEL_ENV[provider]
    model = _strip(os.getenv(env_key)) or default
    if not model:
        raise ValueError(
            f"No model configured for provider {provider!r}. "
            f"Set {env_key} or LLM_MODEL in .env."
        )
    return model


def resolve_ollama_base_url(provider: str) -> str:
    if provider not in PROVIDER_BASE_URL_ENV:
        raise ValueError(f"Not an Ollama provider: {provider}")
    env_key, default = PROVIDER_BASE_URL_ENV[provider]
    base_url = _strip(os.getenv(env_key)) or default
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def model_env_key(provider: str) -> str:
    """Env var name that sets the model for this provider (for docs/logging)."""
    return PROVIDER_MODEL_ENV[provider.lower()][0]


def resolve_provider_base_url(provider: str) -> str:
    if provider not in PROVIDER_BASE_URL_ENV:
        raise ValueError(f"No configurable base URL for provider: {provider}")
    env_key, default = PROVIDER_BASE_URL_ENV[provider]
    return (_strip(os.getenv(env_key)) or default).rstrip("/")
