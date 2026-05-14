import os
import logging
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider="groq"):
        self.provider = provider.lower()
        self.client = None
        self.model = None
        self._setup_client()

    def _setup_client(self):
        if self.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment.")
            # Groq provides an OpenAI compatible endpoint
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key
            )
            self.model = "llama3-70b-8192" # Example model
            
        elif self.provider == "cerebras":
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                logger.warning("CEREBRAS_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key
            )
            self.model = "llama3.1-70b" # Example model
            
        elif self.provider == "ollama":
            # Ollama provides an OpenAI compatible endpoint locally
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama" # required but ignored
            )
            self.model = "llama3" # Example local model
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_response(self, messages, tools=None):
        """
        Calls the LLM with the given messages and optional tools.
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0, # Deterministic for decision making
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            logger.error(f"Error calling LLM ({self.provider}): {e}")
            return None
