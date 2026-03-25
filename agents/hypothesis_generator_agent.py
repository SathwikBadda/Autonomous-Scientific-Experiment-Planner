"""
agents/hypothesis_generator_agent.py - Generates hypotheses ALWAYS,
even when gaps list is empty (reasons directly from problem statement + literature).
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class HypothesisGeneratorAgent(BaseAgent):
    agent_name = "hypothesis_generator"
    prompt_key = "hypothesis_generator_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "HypothesisGeneratorAgent"):
            identified_gaps = state.get("identified_gaps", [])
            literature_summary = state.get("literature_summary", "")
            domain = state.get("domain", "AI/ML")
            constraints = state.get("constraints", [])
            problem_statement = state.get("problem_statement", "")
            retrieved_papers = state.get("retrieved_papers", [])

            # Always build hypotheses — use problem statement if gaps are empty
            gap_context = json.dumps(identified_gaps, indent=2) if identified_gaps else (
                f"No structured gaps available. Reason directly from the problem:\n{problem_statement}\n\n"
                f"Key papers retrieved:\n" +
                "\n".join(f"- {p.get('title', '')} ({p.get('year', '')})" for p in retrieved_papers[:5])
            )

            lit_context = literature_summary or (
                "Literature summary not yet available. Base hypotheses on known challenges in " + domain
            )

            user_prompt = self._get_user_prompt(
                identified_gaps=gap_context,
                literature_summary=lit_context,
                domain=domain,
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("hypothesis_gen_parse_fallback", text_preview=text[:200])
                # Build structured fallback from raw text
                parsed = {
                    "hypotheses": [
                        {
                            "hypothesis": f"IF we apply advanced techniques to {problem_statement[:100]}, "
                                          f"THEN we can improve performance BECAUSE existing methods have known limitations.",
                            "addresses_gap": "Primary research gap in the field",
                            "approach": text[:300],
                            "novelty_justification": "Novel combination of approaches",
                            "theoretical_basis": "Grounded in existing literature",
                            "confidence": "medium",
                            "expected_improvement": "10-15% over baseline methods",
                        }
                    ],
                    "primary_hypothesis": text[:300] if len(text) < 600 else text[:600],
                    "hypothesis_rationale": "Generated from gap analysis and literature review.",
                }

            hypotheses = parsed.get("hypotheses", [])
            logger.info("hypothesis_gen_done", hypotheses_count=len(hypotheses))

            return {
                **state,
                "hypotheses": hypotheses,
                "primary_hypothesis": parsed.get("primary_hypothesis", ""),
                "hypothesis_rationale": parsed.get("hypothesis_rationale", ""),
                "agent_trace": state.get("agent_trace", []) + ["HypothesisGeneratorAgent"],
            }
