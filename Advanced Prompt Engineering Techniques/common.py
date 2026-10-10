"""Shared OpenAI client setup for the prompting examples."""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def run_prompt(messages: Any, *, response_format: Any | None = None) -> str:
    """Send chat messages to OpenAI and return the assistant's text."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise SystemExit("Set OPENAI_API_KEY in a .env file in the workspace root.")

    client = OpenAI(api_key=api_key)
    request: dict[str, Any] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
    }
    if response_format is not None:
        request["response_format"] = response_format

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or "(No text response)"
