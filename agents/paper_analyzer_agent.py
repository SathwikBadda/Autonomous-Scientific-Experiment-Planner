"""
agents/paper_analyzer_agent.py - Extracts structured knowledge from papers.
Node 3 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from rag.pipeline import build_rag_context
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class PaperAnalyzerAgent(BaseAgent):
    """
    Analyzes retrieved papers and extracts:
    - Core methods and innovations
    - Results and metrics
    - Stated + implicit limitations
    - Reproducibility signals
    """

    agent_name = "paper_analyzer"
    prompt_key = "paper_analyzer_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "PaperAnalyzerAgent"):
            papers = state.get("retrieved_papers", [])
            problem_statement = state.get("problem_statement", "")

            if not papers:
                logger.warning("no_papers_for_analysis")
                return {
                    **state,
                    "paper_analyses": [],
                    "common_methods": [],
                    "common_datasets": [],
                    "performance_frontier": "No papers available for analysis.",
                    "literature_summary": "No papers retrieved.",
                    "agent_trace": state.get("agent_trace", []) + ["PaperAnalyzerAgent"],
                }

            papers_text = build_rag_context(papers, max_papers=min(len(papers), 8))

            user_prompt = self._get_user_prompt(
                papers=papers_text,
                problem_statement=problem_statement,
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("paper_analyzer_parse_fallback")
                parsed = {
                    "paper_analyses": [],
                    "common_methods": [],
                    "common_datasets": [],
                    "performance_frontier": text[:500],
                }

            # Build literature summary for downstream agents
            analyses = parsed.get("paper_analyses", [])
            summary_parts = []
            for a in analyses[:5]:
                title = a.get("title", "Unknown")
                method = a.get("core_method", "")
                best = a.get("best_results", "")
                summary_parts.append(f"• {title}: {method}. Results: {best}")

            literature_summary = (
                f"Analysis of {len(analyses)} papers:\n" + "\n".join(summary_parts)
            )

            logger.info(
                "paper_analyzer_done",
                papers_analyzed=len(analyses),
                common_methods=len(parsed.get("common_methods", [])),
            )

            return {
                **state,
                "paper_analyses": analyses,
                "common_methods": parsed.get("common_methods", []),
                "common_datasets": parsed.get("common_datasets", []),
                "performance_frontier": parsed.get("performance_frontier", ""),
                "literature_summary": literature_summary,
                "agent_trace": state.get("agent_trace", []) + ["PaperAnalyzerAgent"],
            }
