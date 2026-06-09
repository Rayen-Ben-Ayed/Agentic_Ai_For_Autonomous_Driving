import logging
import os
import re
import time

import httpx
from openai import OpenAI

from agent.gemini_adapter import (
    GeminiChatMessage,
    messages_to_gemini,
    openai_tools_to_gemini,
    parse_gemini_response,
)
from agent.llm_config import (
    model_env_key,
    resolve_gemini_base_url,
    resolve_model,
    resolve_ollama_base_url,
    resolve_provider,
)
from pipeline_log import log_stage

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = resolve_provider(provider)
        self.client: OpenAI | None = None
        self._gemini_api_key: str | None = None
        self._gemini_base_url: str | None = None
        self._gemini_timeout: float = 20.0
        self.model: str | None = None
        self._setup_client()

    def _setup_client(self) -> None:
        self.model = resolve_model(self.provider)
        timeout = float(os.getenv("LLM_TIMEOUT", "20.0"))

        if self.provider == "gemini":
            self._gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not self._gemini_api_key:
                logger.warning("GEMINI_API_KEY not found in environment.")
            self._gemini_base_url = resolve_gemini_base_url()
            self._gemini_timeout = float(os.getenv("GEMINI_TIMEOUT", str(timeout)))

        elif self.provider == "groq":
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

    def _generate_gemini_response(
        self, messages, tools=None
    ) -> GeminiChatMessage | None:
        system_instruction, contents = messages_to_gemini(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = openai_tools_to_gemini(tools)
            payload["toolConfig"] = {
                "functionCallingConfig": {"mode": "AUTO"},
            }

        url = f"{self._gemini_base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._gemini_api_key or "",
        }

        with httpx.Client(timeout=self._gemini_timeout) as http:
            response = http.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return parse_gemini_response(response.json())

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

                if self.provider == "gemini":
                    msg = self._generate_gemini_response(messages, tools=tools)
                else:
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

                if msg.tool_calls:
                    called = [tc.function.name for tc in msg.tool_calls]
                    log_stage(logger, "LLM", "response tool_calls=%s", called)
                else:
                    preview = (msg.content or "")[:80]
                    log_stage(logger, "LLM", "response text=%r", preview)
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
