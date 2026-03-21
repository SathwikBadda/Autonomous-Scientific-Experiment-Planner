"""
agents/hypothesis_generator_agent.py - Generates novel, testable hypotheses.
Node 5 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class HypothesisGeneratorAgent(BaseAgent):
    """
    Generates novel, falsifiable research hypotheses grounded in identified gaps.
    Each hypothesis follows the IF...THEN...BECAUSE format.
    """

    agent_name = "hypothesis_generator"
    prompt_key = "hypothesis_generator_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "HypothesisGeneratorAgent"):
            identified_gaps = state.get("identified_gaps", [])
            literature_summary = state.get("literature_summary", "")
            domain = state.get("domain", "AI/ML")
            constraints = state.get("constraints", [])

            if not identified_gaps:
                logger.warning("no_gaps_for_hypothesis")
                return {
                    **state,
                    "hypotheses": [],
                    "primary_hypothesis": "",
                    "hypothesis_rationale": "No gaps available for hypothesis generation.",
                    "agent_trace": state.get("agent_trace", []) + ["HypothesisGeneratorAgent"],
                }

            user_prompt = self._get_user_prompt(
                identified_gaps=json.dumps(identified_gaps, indent=2),
                literature_summary=literature_summary,
                domain=domain,
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("hypothesis_gen_parse_fallback")
                parsed = {
                    "hypotheses": [],
                    "primary_hypothesis": text[:300],
                    "hypothesis_rationale": "Generated from gap analysis.",
                }

            hypotheses = parsed.get("hypotheses", [])
            min_h = self._agent_cfg.get("min_hypotheses", 3)
            if len(hypotheses) < min_h:
                logger.warning(
                    "insufficient_hypotheses", found=len(hypotheses), min=min_h
                )

            logger.info(
                "hypothesis_gen_done",
                hypotheses_count=len(hypotheses),
                primary=parsed.get("primary_hypothesis", "")[:80],
            )

            return {
                **state,
                "hypotheses": hypotheses,
                "primary_hypothesis": parsed.get("primary_hypothesis", ""),
                "hypothesis_rationale": parsed.get("hypothesis_rationale", ""),
                "agent_trace": state.get("agent_trace", []) + ["HypothesisGeneratorAgent"],
            }
