"""
agents/base_agent.py - Base class for all agents.
Provides: Claude API calls, JSON parsing, retry logic, Langfuse logging.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import config, get_anthropic_key, prompts
from utils.logger import get_logger
from utils.tracer import agent_span, log_llm_call

logger = get_logger(__name__)


class BaseAgent:
    """
    Base agent providing LLM call infrastructure.
    All specialized agents inherit from this class.
    """

    agent_name: str = "base_agent"
    prompt_key: str = ""  # key in agent_prompts.yaml

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=get_anthropic_key())
        self._model = config["llm"]["model"]
        self._max_tokens = config["llm"]["max_tokens"]
        self._temperature = config["llm"]["temperature"]
        self._agent_cfg = config.get("agents", {}).get(
            self.agent_name.replace(" ", "_"), {}
        )
        self._prompts = prompts.get(self.prompt_key, {})

    def _get_system_prompt(self) -> str:
        return self._prompts.get("system", "You are a helpful AI research assistant.")

    def _get_user_prompt(self, **kwargs) -> str:
        template = self._prompts.get("user", "{input}")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("prompt_format_missing_key", key=str(e), agent=self.agent_name)
            return template

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
    )
    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        messages: Optional[list] = None,
        trace=None,
    ) -> "anthropic.types.Message":
        """Call Claude API with optional tools and conversation history."""
        sys_prompt = system_prompt or self._get_system_prompt()

        if messages is None:
            messages = [{"role": "user", "content": user_prompt}]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": sys_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"}

        logger.info(
            "llm_call",
            agent=self.agent_name,
            model=self._model,
            has_tools=bool(tools),
        )
        response = self._client.messages.create(**kwargs)

        if trace:
            log_llm_call(
                trace_or_span=trace,
                model=self._model,
                prompt=user_prompt[:3000],
                response=self._extract_text(response),
                agent_name=self.agent_name,
            )
        return response

    def _extract_text(self, response: anthropic.types.Message) -> str:
        """Extract all text blocks from a Claude response."""
        texts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]
        return "\n".join(texts)

    def _extract_tool_use(
        self, response: anthropic.types.Message
    ) -> list[tuple[str, dict]]:
        """Extract tool_use blocks: returns list of (tool_name, tool_input)."""
        return [
            (block.name, block.input)
            for block in response.content
            if block.type == "tool_use"
        ]

    def _parse_json(self, text: str) -> dict:
        """
        Robustly extract JSON from LLM response text.
        Handles markdown code fences and trailing text.
        """
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` blocks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(1) if "```" in pattern else match.group(0)
                try:
                    return json.loads(candidate.strip())
                except json.JSONDecodeError:
                    continue

        logger.error(
            "json_parse_failed",
            agent=self.agent_name,
            raw_text=text[:300],
        )
        raise ValueError(f"[{self.agent_name}] Could not parse JSON from LLM response")

    def run(self, state: dict, trace=None) -> dict:
        """Override in subclasses. Must return updated state dict."""
        raise NotImplementedError
