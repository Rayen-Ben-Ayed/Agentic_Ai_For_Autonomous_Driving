import os
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from pipeline_log import log_stage

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider="groq"):
        self.provider = provider.lower()
        self.client = None
        self.model = None
        self._setup_client()

    def _setup_client(self):
        # A single LLM_MODEL override wins for any provider; otherwise fall back
        # to a provider-specific default (also overridable per provider).
        model_override = os.environ.get("LLM_MODEL")

        if self.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment.")
            # Groq provides an OpenAI compatible endpoint
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                max_retries=1,
                timeout=20.0,
            )
            self.model = model_override or os.environ.get(
                "GROQ_MODEL", "openai/gpt-oss-120b"
            )
 #self.model = "llama-3.3-70b-versatile" # Updated model
            #self.model = "qwen/qwen3-32b" # Updated model
            #self.model = "openai/gpt-oss-120b" # Updated model
            #self.model = "llama-3.1-8b-instant"
        
        elif self.provider == "cerebras":
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                logger.warning("CEREBRAS_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key,
                max_retries=1,
                timeout=20.0,
            )
            self.model = model_override or os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")

        elif self.provider == "ollama":
            # Ollama provides an OpenAI compatible endpoint locally
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama" # required but ignored
            )
            self.model = model_override or os.environ.get("OLLAMA_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_response(self, messages, tools=None):
        """
        Calls the LLM with the given messages and optional tools.
        """
        try:
            tool_names = [t["function"]["name"] for t in tools] if tools else []
            log_stage(
                logger,
                "LLM",
                "request model=%s msgs=%d tools=%s",
                self.model,
                len(messages),
                tool_names or "none",
            )
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
            logger.error(f"Error calling LLM ({self.provider}): {e}")
            return None
