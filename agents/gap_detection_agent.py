"""
agents/gap_detection_agent.py - Identifies research gaps from paper analyses.
Node 4 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class GapDetectionAgent(BaseAgent):
    """
    Critical analysis agent that identifies what is MISSING in the literature.
    Outputs structured gaps with evidence, type, and difficulty rating.
    """

    agent_name = "gap_detection"
    prompt_key = "gap_detection_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "GapDetectionAgent"):
            paper_analyses = state.get("paper_analyses", [])
            research_scope = state.get("research_scope", {})
            constraints = state.get("constraints", [])

            if not paper_analyses:
                logger.warning("no_analyses_for_gap_detection")
                return {
                    **state,
                    "identified_gaps": [],
                    "priority_gaps": [],
                    "gap_summary": "Insufficient literature for gap detection.",
                    "agent_trace": state.get("agent_trace", []) + ["GapDetectionAgent"],
                }

            user_prompt = self._get_user_prompt(
                paper_analyses=json.dumps(paper_analyses, indent=2),
                research_scope=json.dumps(research_scope, indent=2),
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("gap_detection_parse_fallback")
                parsed = {
                    "identified_gaps": [],
                    "priority_gaps": [],
                    "gap_summary": text[:300],
                }

            gaps = parsed.get("identified_gaps", [])
            min_gaps = self._agent_cfg.get("min_gaps", 3)
            if len(gaps) < min_gaps:
                logger.warning(
                    "insufficient_gaps",
                    found=len(gaps),
                    min=min_gaps,
                )

            logger.info(
                "gap_detection_done",
                gaps_found=len(gaps),
                priority_gaps=len(parsed.get("priority_gaps", [])),
            )

            return {
                **state,
                "identified_gaps": gaps,
                "priority_gaps": parsed.get("priority_gaps", []),
                "gap_summary": parsed.get("gap_summary", ""),
                "agent_trace": state.get("agent_trace", []) + ["GapDetectionAgent"],
            }
