"""
agents/experiment_planner_agent.py - Designs detailed experiment protocols.
Node 6 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class ExperimentPlannerAgent(BaseAgent):
    """
    Translates hypotheses into concrete, rigorous experiment plans.
    Outputs: architecture, training config, baselines, metrics, ablations.
    """

    agent_name = "experiment_planner"
    prompt_key = "experiment_planner_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "ExperimentPlannerAgent"):
            primary_hypothesis = state.get("primary_hypothesis", "")
            hypotheses = state.get("hypotheses", [])
            performance_frontier = state.get("performance_frontier", "")
            constraints = state.get("constraints", [])

            user_prompt = self._get_user_prompt(
                primary_hypothesis=primary_hypothesis,
                hypotheses=json.dumps(hypotheses, indent=2),
                performance_frontier=performance_frontier,
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("experiment_planner_parse_fallback")
                parsed = {
                    "experiment_plan": {
                        "model_architecture": "To be determined",
                        "key_components": [],
                        "training_strategy": {},
                        "baseline_models": [],
                        "evaluation_metrics": [],
                        "ablation_studies": [],
                        "success_criteria": "",
                        "estimated_compute": "Unknown",
                        "timeline_weeks": 12,
                    }
                }

            experiment_plan = parsed.get("experiment_plan", parsed)

            logger.info(
                "experiment_planner_done",
                baselines=len(experiment_plan.get("baseline_models", [])),
                metrics=len(experiment_plan.get("evaluation_metrics", [])),
            )

            return {
                **state,
                "experiment_plan": experiment_plan,
                "agent_trace": state.get("agent_trace", []) + ["ExperimentPlannerAgent"],
            }
