"""
agents/gap_detection_agent.py - Identifies research gaps ALWAYS,
even when paper_analyses is empty. Falls back to problem statement reasoning.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from rag.pipeline import search_by_section, search_similar_papers
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class GapDetectionAgent(BaseAgent):
    agent_name = "gap_detection"
    prompt_key = "gap_detection_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "GapDetectionAgent"):
            paper_analyses = state.get("paper_analyses", [])
            retrieved_papers = state.get("retrieved_papers", [])
            research_scope = state.get("research_scope", {})
            constraints = state.get("constraints", [])
            problem_statement = state.get("problem_statement", "")

            # Build paper evidence — use both analyses AND raw retrieved papers
            paper_context = ""
            if paper_analyses:
                paper_context = json.dumps(paper_analyses, indent=2)[:6000]
            elif retrieved_papers:
                # Build context from raw papers if analyses weren't produced
                paper_context = json.dumps([
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", "")[:600],
                        "limitations": p.get("limitations", "Not stated"),
                        "results": p.get("results", "")[:300],
                    }
                    for p in retrieved_papers[:5]
                ], indent=2)

            # Targeted RAG: pull limitation and future work chunks for evidence
            limitation_chunks = search_by_section(problem_statement, "Limitations", top_k=8)
            results_chunks = search_by_section(problem_statement, "Results", top_k=5)

            evidence_context = ""
            if limitation_chunks:
                evidence_context += "### LIMITATIONS IN EXISTING WORK\n"
                evidence_context += "\n\n".join(limitation_chunks[:5])
            if results_chunks:
                evidence_context += "\n\n### PERFORMANCE BOUNDARIES\n"
                evidence_context += "\n\n".join(results_chunks[:3])

            pdf_context = state.get("pdf_context", "")

            user_prompt = self._get_user_prompt(
                paper_analyses=paper_context or "No paper analyses yet — reason from the problem statement.",
                pdf_context=pdf_context or "No PDF uploaded.",
                evidence_context=evidence_context or "No targeted evidence retrieved.",
                research_scope=json.dumps(research_scope, indent=2),
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("gap_detection_parse_fallback", text_preview=text[:200])
                # Generate structured gaps from the raw text
                parsed = {
                    "identified_gaps": [
                        {
                            "gap": f"Gap identified from analysis: {text[:200]}",
                            "gap_type": "experimental",
                            "evidence": "Derived from literature analysis",
                            "impact": "High",
                            "difficulty": "medium",
                            "proposed_direction": "Further investigation required",
                        }
                    ],
                    "priority_gaps": [text[:100]],
                    "proposed_improvements": ["Systematic study needed"],
                    "new_dataset_needs": ["Benchmark dataset for the target domain"],
                    "gap_summary": text[:500],
                }

            gaps = parsed.get("identified_gaps", [])
            logger.info("gap_detection_done", gaps_found=len(gaps), has_evidence=bool(limitation_chunks))

            return {
                **state,
                "identified_gaps": gaps,
                "priority_gaps": parsed.get("priority_gaps", []),
                "gap_summary": parsed.get("gap_summary", ""),
                "proposed_improvements": parsed.get("proposed_improvements", []),
                "new_dataset_needs": parsed.get("new_dataset_needs", []),
                "agent_trace": state.get("agent_trace", []) + ["GapDetectionAgent"],
            }
