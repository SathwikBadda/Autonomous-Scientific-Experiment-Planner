"""
agents/critic_agent.py - Peer-review quality gate with iterative refinement.
Node 8 (final) in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from config.settings import config
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """
    Scientific peer-review agent that:
    1. Critiques the full research plan
    2. Assigns novelty/feasibility/impact scores
    3. Provides revision suggestions
    4. Produces final expected outcomes
    """

    agent_name = "critic"
    prompt_key = "critic_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "CriticAgent"):
            user_prompt = self._get_user_prompt(
                problem_statement=state.get("problem_statement", ""),
                literature_summary=state.get("literature_summary", ""),
                identified_gaps=json.dumps(state.get("identified_gaps", []), indent=2),
                hypotheses=json.dumps(state.get("hypotheses", []), indent=2),
                experiment_plan=json.dumps(state.get("experiment_plan", {}), indent=2),
                datasets=json.dumps(state.get("datasets", []), indent=2),
                constraints=json.dumps(state.get("constraints", []), indent=2),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("critic_parse_fallback")
                parsed = {
                    "critique": {"strengths": [], "weaknesses": [], "revised_suggestions": {}},
                    "scores": {
                        "novelty_score": 5,
                        "novelty_reasoning": "N/A",
                        "feasibility_score": 5,
                        "feasibility_reasoning": "N/A",
                        "impact_score": 5,
                        "impact_reasoning": "N/A",
                    },
                    "overall_recommendation": "revise",
                    "expected_outcomes": text[:300],
                }

            scores = parsed.get("scores", {})
            novelty = scores.get("novelty_score", 0)
            feasibility = scores.get("feasibility_score", 0)
            impact = scores.get("impact_score", 0)

            # Composite score
            score_cfg = config.get("scoring", {})
            composite = (
                novelty * score_cfg.get("novelty_weight", 0.4)
                + feasibility * score_cfg.get("feasibility_weight", 0.35)
                + impact * score_cfg.get("impact_weight", 0.25)
            )

            logger.info(
                "critic_done",
                novelty=novelty,
                feasibility=feasibility,
                impact=impact,
                composite=round(composite, 2),
                recommendation=parsed.get("overall_recommendation"),
            )

            return {
                **state,
                "critique": parsed.get("critique", {}),
                "novelty_score": novelty,
                "feasibility_score": feasibility,
                "impact_score": impact,
                "composite_score": round(composite, 2),
                "overall_recommendation": parsed.get("overall_recommendation", "revise"),
                "expected_outcomes": parsed.get("expected_outcomes", ""),
                "agent_trace": state.get("agent_trace", []) + ["CriticAgent"],
            }
