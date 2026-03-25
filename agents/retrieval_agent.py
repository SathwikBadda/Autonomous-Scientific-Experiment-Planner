"""
agents/retrieval_agent.py - Retrieves papers via MCP-style arXiv tool calling.
After retrieval: saves papers to session folder, chunks by section, indexes in FAISS.
Node 2 in the LangGraph pipeline.
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from agents.base_agent import BaseAgent
from config.settings import config
from rag.pipeline import index_papers, reset_index, search_similar_papers
from rag.session_manager import save_all_papers
from tools.arxiv_tool import dispatch_tool_call, get_all_tool_schemas
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6  # max agentic loops


class RetrievalAgent(BaseAgent):
    """
    MCP-style retrieval agent. Claude autonomously calls search_arxiv
    for each search query, then papers are:
      1. Saved to session folder as JSON + Markdown
      2. Chunked by section (Abstract/Methods/Results/Limitations)
      3. Indexed into FAISS for targeted RAG queries
    """

    agent_name = "retrieval"
    prompt_key = "retrieval_agent"

    def _agentic_retrieval_loop(
        self, search_queries: list, research_scope: dict, trace=None
    ) -> list:
        """
        Agentic tool-use loop:
        1. Send queries + tools to Claude
        2. Claude returns tool_use blocks
        3. Execute tools, return results to Claude
        4. Repeat until Claude stops calling tools
        """
        all_papers: list[dict] = []
        seen_ids: set = set()

        queries_str = "\n".join(f"- {q}" for q in search_queries)
        user_prompt = self._get_user_prompt(
            research_scope=json.dumps(research_scope, indent=2),
            search_queries=queries_str,
        )

        messages = [{"role": "user", "content": user_prompt}]
        tools = get_all_tool_schemas()

        for round_num in range(MAX_TOOL_ROUNDS):
            logger.info("retrieval_round", round=round_num + 1)

            response = self._call_llm(
                user_prompt=user_prompt,
                tools=tools,
                messages=messages,
                trace=trace,
            )

            tool_uses = self._extract_tool_use(response)
            text_blocks = self._extract_text(response)

            if not tool_uses:
                logger.info("retrieval_no_more_tools", round=round_num + 1)
                break

            # Build assistant message with all content blocks
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

            # Execute all tool calls and collect results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                logger.info("tool_call", tool=tool_name, input=str(tool_input)[:200])
                result = dispatch_tool_call(tool_name, tool_input)

                # Deduplicate papers
                for paper in result.get("papers", []):
                    pid = paper.get("arxiv_id", paper.get("title", ""))
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        all_papers.append(paper)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        logger.info("retrieval_complete", total_papers=len(all_papers))
        return all_papers

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "RetrievalAgent", input_data=state.get("search_queries")):
            search_queries = state.get("search_queries", [])
            research_scope = state.get("research_scope", {})
            problem_statement = state.get("problem_statement", "")
            session = state.get("session", {})

            if not search_queries:
                search_queries = [problem_statement]

            # Clear previous FAISS index for fresh run
            reset_index()

            # Agentic retrieval
            papers = self._agentic_retrieval_loop(
                search_queries, research_scope, trace=trace
            )

            # 1. Save papers to session folder (JSON + Markdown)
            session_index_dir = session.get("index_dir") if session else None
            if session:
                try:
                    save_all_papers(session, papers)
                    logger.info("session_papers_saved", count=len(papers))
                except Exception as e:
                    logger.warning("session_save_failed", error=str(e))

            # 2. Chunk by section and index into ChromaDB (persists to session/index/)
            n_indexed = index_papers(papers, session_index_dir=session_index_dir)
            logger.info("paper_chunks_indexed_chroma", count=n_indexed)

            # 3. RAG: retrieve top-10, then focus agents on top-3 most relevant
            all_relevant = search_similar_papers(query=problem_statement, top_k=10)
            top_papers = all_relevant[:3]  # agents reason deeply on top-3 only

            # 4. Full PDF download + section detection for top-3 papers
            papers_dir = session.get("papers_dir", "") if session else ""
            if papers_dir and top_papers:
                try:
                    from rag.paper_downloader import process_paper_to_markdown_and_chunks
                    deep_chunks = []
                    for paper in top_papers:
                        result = process_paper_to_markdown_and_chunks(paper, papers_dir)
                        if result.get("success"):
                            deep_chunks.extend(result.get("chunks", []))
                            # Update paper dict with extracted sections
                            paper.update(result.get("sections", {}))
                            logger.info(
                                "deep_paper_processed",
                                arxiv_id=paper.get("arxiv_id", "?"),
                                chunks=len(result.get("chunks", [])),
                            )

                    # Re-index with deep chunks into ChromaDB
                    if deep_chunks:
                        chunk_texts = [c["chunk_text"] for c in deep_chunks]
                        index_chunks(chunk_texts, session_index_dir=session_index_dir)
                        logger.info("deep_chunks_indexed", count=len(deep_chunks))
                except Exception as e:
                    logger.warning("deep_paper_processing_failed", error=str(e))

            logger.info(
                "retrieval_agent_done",
                total_fetched=len(papers),
                top_3_titles=[p.get("title", "?")[:60] for p in top_papers],
                chunks_indexed=n_indexed,
            )


            return {
                **state,
                "raw_papers": papers,
                "retrieved_papers": top_papers,
                "total_papers_fetched": len(papers),
                "agent_trace": state.get("agent_trace", []) + ["RetrievalAgent"],
            }
