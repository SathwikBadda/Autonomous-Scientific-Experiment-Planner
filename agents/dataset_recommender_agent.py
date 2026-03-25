"""
agents/dataset_recommender_agent.py - Recommends datasets using HuggingFace tool + LLM reasoning.
Node 7 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from tools.arxiv_tool import dispatch_tool_call, get_all_tool_schemas
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 3


class DatasetRecommenderAgent(BaseAgent):
    """
    Recommends appropriate open-access datasets using the search_datasets tool.
    Claude autonomously searches HuggingFace for relevant datasets, then reasons
    about which are best suited to evaluate the proposed experiment.
    """

    agent_name = "dataset_recommender"
    prompt_key = "dataset_recommender_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "DatasetRecommenderAgent"):
            experiment_plan = state.get("experiment_plan", {})
            domain = state.get("domain", "AI/ML")
            common_datasets = state.get("common_datasets", [])
            constraints = state.get("constraints", [])
            problem_statement = state.get("problem_statement", "")
            hypotheses = state.get("hypotheses", [])

            # Agentic loop: let Claude call search_datasets tool
            dataset_search_results = []
            tools = get_all_tool_schemas()

            user_prompt = self._get_user_prompt(
                experiment_plan=json.dumps(experiment_plan, indent=2),
                domain=domain,
                common_datasets=json.dumps(common_datasets),
                constraints=json.dumps(constraints),
            )

            messages = [{"role": "user", "content": user_prompt}]

            for round_num in range(MAX_TOOL_ROUNDS):
                response = self._call_llm(
                    user_prompt=user_prompt,
                    tools=tools,
                    messages=messages,
                    trace=trace,
                )
                tool_uses = self._extract_tool_use(response)
                text_blocks = self._extract_text(response)

                if not tool_uses:
                    break

                # Build messages
                assistant_content = []
                if text_blocks:
                    assistant_content.append({"type": "text", "text": text_blocks})
                for block in response.content:
                    if block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = dispatch_tool_call(block.name, block.input)
                    if block.name == "search_datasets":
                        dataset_search_results.extend(result.get("datasets", []))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                messages.append({"role": "user", "content": tool_results})

            # Final synthesis call (no tools)
            synth_prompt = (
                f"Based on the datasets found and experiment plan, return a JSON with:\n"
                f"{{\n"
                f'  "datasets": [{{"name": "...", "url": "...", "description": "...", "why_suitable": "..."}}],\n'
                f'  "data_collection_needed": true/false,\n'
                f'  "data_collection_plan": "..."\n'
                f"}}\n\n"
                f"Dataset search results:\n{json.dumps(dataset_search_results, indent=2)[:3000]}\n\n"
                f"Problem: {problem_statement}\n"
                f"Existing datasets in literature: {json.dumps(common_datasets)}"
            )
            synth_response = self._call_llm(user_prompt=synth_prompt, trace=trace)
            text = self._extract_text(synth_response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("dataset_recommender_parse_fallback")
                # Fallback: convert raw search results
                parsed = {
                    "datasets": [
                        {"name": d["name"], "url": d.get("url", ""), "description": d.get("description", ""),
                         "why_suitable": "Relevant based on search"}
                        for d in dataset_search_results[:5]
                    ],
                    "data_collection_needed": False,
                    "data_collection_plan": "",
                }

            datasets = parsed.get("datasets", [])
            logger.info("dataset_recommender_done", datasets_found=len(datasets), searched=len(dataset_search_results))

            return {
                **state,
                "datasets": datasets,
                "data_collection_needed": parsed.get("data_collection_needed", False),
                "data_collection_plan": parsed.get("data_collection_plan", ""),
                "agent_trace": state.get("agent_trace", []) + ["DatasetRecommenderAgent"],
            }
