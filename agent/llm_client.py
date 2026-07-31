import logging
import os
import re
import time

from openai import OpenAI

from agent.llm_config import (
    model_env_key,
    resolve_model,
    resolve_ollama_base_url,
    resolve_provider,
    resolve_provider_base_url,
)
from pipeline_log import log_stage

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: str | None = None, benchmark_collector=None):
        self.provider = resolve_provider(provider)
        self._benchmark_collector = benchmark_collector
        self.client: OpenAI | None = None
        self.model: str | None = None
        self._verbose = os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"
        self._setup_client()

    def _setup_client(self) -> None:
        self.model = resolve_model(self.provider)
        timeout = float(os.getenv("LLM_TIMEOUT", "20.0"))

        if self.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                max_retries=1,
                timeout=timeout,
            )

        elif self.provider == "cerebras":
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                logger.warning("CEREBRAS_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key,
                max_retries=1,
                timeout=timeout,
            )

        elif self.provider in ("ollama", "ollama-remote"):
            base_url = resolve_ollama_base_url(self.provider)
            ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama",
                max_retries=1,
                timeout=ollama_timeout,
            )

        elif self.provider == "academic_cloud":
            api_key = os.environ.get("ACADEMIC_CLOUD_API_KEY")
            if not api_key:
                logger.warning("ACADEMIC_CLOUD_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url=resolve_provider_base_url("academic_cloud"),
                api_key=api_key,
                max_retries=1,
                timeout=timeout,
            )

        log_stage(
            logger,
            "LLM",
            "provider=%s model=%s env=%s",
            self.provider,
            self.model,
            model_env_key(self.provider),
        )

    @staticmethod
    def _rate_limit_wait_s(error: Exception) -> float | None:
        text = str(error)
        if "429" not in text and "rate_limit" not in text.lower():
            return None
        match = re.search(r"try again in ([0-9.]+)s", text, re.I)
        if match:
            return float(match.group(1)) + 0.25
        return float(os.getenv("LLM_RATE_LIMIT_BACKOFF_S", "6.0"))

    def generate_response(self, messages, tools=None):
        """Calls the LLM with the given messages and optional tools."""
        max_retries = int(os.getenv("LLM_RATE_LIMIT_RETRIES", "3"))
        tool_names = [t["function"]["name"] for t in tools] if tools else []

        for attempt in range(1, max_retries + 1):
            try:
                log_stage(
                    logger,
                    "LLM",
                    "request model=%s msgs=%d tools=%s",
                    self.model,
                    len(messages),
                    tool_names or "none",
                )

                started = time.perf_counter()
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                finish_reason = None
                usage = None

                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                try:
                    finish_reason = response.choices[0].finish_reason
                    if response.usage is not None:
                        prompt_tokens = response.usage.prompt_tokens
                        completion_tokens = response.usage.completion_tokens
                        total_tokens = response.usage.total_tokens
                        usage = (
                            f"prompt={prompt_tokens} "
                            f"completion={completion_tokens} "
                            f"total={total_tokens}"
                        )
                except (AttributeError, IndexError):
                    pass

                latency_ms = (time.perf_counter() - started) * 1000
                if self._benchmark_collector is not None:
                    self._benchmark_collector.record_llm_call(
                        latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

                if msg.tool_calls:
                    called = [tc.function.name for tc in msg.tool_calls]
                    log_stage(
                        logger,
                        "LLM",
                        "response tool_calls=%s finish=%s usage=[%s]",
                        called,
                        finish_reason,
                        usage or "n/a",
                    )
                    # The model's chain-of-thought / rationale is otherwise dropped.
                    reasoning = (msg.content or "").strip()
                    if reasoning:
                        log_stage(
                            logger,
                            "LLM",
                            "reasoning: %s",
                            reasoning if self._verbose else reasoning[:300],
                        )
                else:
                    preview = (msg.content or "")[:80]
                    log_stage(
                        logger,
                        "LLM",
                        "response text=%r finish=%s usage=[%s]",
                        preview,
                        finish_reason,
                        usage or "n/a",
                    )
                return msg
            except Exception as e:
                wait_s = self._rate_limit_wait_s(e)
                if wait_s is not None and attempt < max_retries:
                    log_stage(
                        logger,
                        "LLM",
                        "rate limited (attempt %d/%d); retry in %.1fs",
                        attempt,
                        max_retries,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                logger.error("Error calling LLM (%s): %s", self.provider, e)
                return None
        return None
