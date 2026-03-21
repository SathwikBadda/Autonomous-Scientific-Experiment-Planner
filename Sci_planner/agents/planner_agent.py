"""
agents/planner_agent.py - Parses research input and defines scope.
Node 1 in the LangGraph pipeline.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Parses any research input format into a structured research scope.
    Outputs: problem_statement, domain, constraints, and search_queries.
    """

    agent_name = "planner"
    prompt_key = "planner_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "PlannerAgent", input_data=state.get("research_input")):
            research_input = state.get("research_input", "")

            # Normalize input to string
            if isinstance(research_input, dict):
                input_str = json.dumps(research_input, indent=2)
            else:
                input_str = str(research_input)

            user_prompt = self._get_user_prompt(research_input=input_str)
            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("planner_fallback", raw=text[:200])
                parsed = {
                    "problem_statement": input_str,
                    "domain": "AI/ML",
                    "sub_domain": "General",
                    "task": "Research",
                    "constraints": [],
                    "open_challenges": [],
                    "search_queries": [input_str],
                }

            logger.info(
                "planner_done",
                domain=parsed.get("domain"),
                queries=len(parsed.get("search_queries", [])),
            )

            return {
                **state,
                "problem_statement": parsed.get("problem_statement", input_str),
                "domain": parsed.get("domain", "AI/ML"),
                "sub_domain": parsed.get("sub_domain", ""),
                "task": parsed.get("task", ""),
                "constraints": parsed.get("constraints", []),
                "open_challenges": parsed.get("open_challenges", []),
                "search_queries": parsed.get("search_queries", [input_str]),
                "research_scope": parsed,
                "agent_trace": state.get("agent_trace", []) + ["PlannerAgent"],
            }
