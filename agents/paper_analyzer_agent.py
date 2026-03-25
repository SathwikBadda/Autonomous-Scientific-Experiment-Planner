"""
agents/paper_analyzer_agent.py - Extracts structured knowledge from papers.
Uses targeted section-level RAG queries to surface limitations and metrics.
Node 3 in the LangGraph pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from rag.pipeline import build_rag_context, search_by_section, search_similar_papers
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class PaperAnalyzerAgent(BaseAgent):
    """
    Analyzes retrieved papers and extracts via targeted RAG:
    - Core methods and innovations
    - Results and evaluation metrics
    - Stated + implicit limitations
    - Datasets used
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

            # Build a rich context: papers + targeted section chunks
            papers_text = build_rag_context(papers=papers, max_items=min(len(papers), 8))

            # Targeted RAG for limitations and evaluation metrics
            limitation_chunks = search_by_section(problem_statement, "Limitations", top_k=5)
            results_chunks = search_by_section(problem_statement, "Results", top_k=5)
            methods_chunks = search_by_section(problem_statement, "Methods", top_k=5)

            targeted_context = ""
            if limitation_chunks:
                targeted_context += "\n\n### LIMITATION EXCERPTS FROM PAPERS\n" + "\n\n".join(limitation_chunks[:3])
            if results_chunks:
                targeted_context += "\n\n### KEY RESULTS & METRICS FROM PAPERS\n" + "\n\n".join(results_chunks[:3])
            if methods_chunks:
                targeted_context += "\n\n### METHODOLOGY EXCERPTS FROM PAPERS\n" + "\n\n".join(methods_chunks[:3])

            user_prompt = self._get_user_prompt(
                papers=papers_text + targeted_context,
                pdf_context=state.get("pdf_context", ""),
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

            # Build rich literature summary for downstream agents
            analyses = parsed.get("paper_analyses", [])
            summary_parts = []
            for a in analyses[:8]:
                title = a.get("title", "Unknown")
                method = a.get("core_method", "")
                best = a.get("best_results", "")
                lim = a.get("limitations", "")
                datasets = ", ".join(a.get("datasets_used", []))
                summary_parts.append(
                    f"• {title}\n"
                    f"  Method: {method}\n"
                    f"  Results: {best}\n"
                    f"  Datasets: {datasets}\n"
                    f"  Limitations: {lim}"
                )

            literature_summary = (
                f"Analyzed {len(analyses)} papers on '{problem_statement}':\n\n"
                + "\n\n".join(summary_parts)
            )

            logger.info(
                "paper_analyzer_done",
                papers_analyzed=len(analyses),
                common_methods=len(parsed.get("common_methods", [])),
                has_limitations=any(a.get("limitations") for a in analyses),
            )

            return {
                **state,
                "paper_analyses": analyses,
                "common_methods": parsed.get("common_methods", []),
                "common_datasets": parsed.get("common_datasets", []),
                "evaluation_metrics_found": parsed.get("evaluation_metrics", []),
                "performance_frontier": parsed.get("performance_frontier", ""),
                "literature_summary": literature_summary,
                "agent_trace": state.get("agent_trace", []) + ["PaperAnalyzerAgent"],
            }
