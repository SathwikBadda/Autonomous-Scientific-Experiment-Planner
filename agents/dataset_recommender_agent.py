"""
agents/dataset_recommender_agent.py - Recommends datasets for experiments.
Node 7 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class DatasetRecommenderAgent(BaseAgent):
    """
    Recommends appropriate open-access datasets for evaluation,
    including out-of-distribution test sets and justifications.
    """

    agent_name = "dataset_recommender"
    prompt_key = "dataset_recommender_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "DatasetRecommenderAgent"):
            experiment_plan = state.get("experiment_plan", {})
            domain = state.get("domain", "AI/ML")
            common_datasets = state.get("common_datasets", [])
            constraints = state.get("constraints", [])

            user_prompt = self._get_user_prompt(
                experiment_plan=json.dumps(experiment_plan, indent=2),
                domain=domain,
                common_datasets=json.dumps(common_datasets),
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("dataset_recommender_parse_fallback")
                parsed = {
                    "datasets": [],
                    "data_collection_needed": False,
                    "data_collection_plan": "",
                }

            datasets = parsed.get("datasets", [])
            logger.info(
                "dataset_recommender_done",
                datasets_found=len(datasets),
                collection_needed=parsed.get("data_collection_needed", False),
            )

            return {
                **state,
                "datasets": datasets,
                "data_collection_needed": parsed.get("data_collection_needed", False),
                "data_collection_plan": parsed.get("data_collection_plan", ""),
                "agent_trace": state.get("agent_trace", []) + ["DatasetRecommenderAgent"],
            }
