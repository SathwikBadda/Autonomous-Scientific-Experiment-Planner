"""
agents/retrieval_agent.py - Retrieves papers via MCP-style arXiv tool calling.
The LLM autonomously decides when and how to call search_arxiv.
Node 2 in the LangGraph pipeline.
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from agents.base_agent import BaseAgent
from config.settings import config
from rag.pipeline import index_papers, reset_index, search_similar_papers
from tools.arxiv_tool import dispatch_tool_call, get_all_tool_schemas
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6  # max agentic loops


class RetrievalAgent(BaseAgent):
    """
    MCP-style retrieval agent. Claude autonomously calls search_arxiv
    for each search query, then papers are indexed into FAISS.
    """

    agent_name = "retrieval"
    prompt_key = "retrieval_agent"

    def _agentic_retrieval_loop(
        self, search_queries: list[str], research_scope: dict, trace=None
    ) -> list[dict]:
        """
        Agentic tool-use loop:
        1. Send queries + tools to Claude
        2. Claude returns tool_use blocks
        3. Execute tools, return results to Claude
        4. Repeat until Claude stops calling tools
        """
        all_papers: list[dict] = []
        seen_ids: set[str] = set()

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

                logger.info("tool_call", tool=tool_name, input=tool_input)
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

            if not search_queries:
                search_queries = [problem_statement]

            # Clear previous FAISS index for fresh run
            reset_index()

            # Agentic retrieval
            papers = self._agentic_retrieval_loop(
                search_queries, research_scope, trace=trace
            )

            # Index into FAISS
            n_indexed = index_papers(papers)
            logger.info("papers_indexed", count=n_indexed)

            # RAG: retrieve top-k most relevant to problem statement
            top_papers = search_similar_papers(
                query=problem_statement,
                top_k=config["rag"]["top_k"],
            )

            logger.info(
                "retrieval_agent_done",
                total_fetched=len(papers),
                top_k=len(top_papers),
            )

            return {
                **state,
                "raw_papers": papers,
                "retrieved_papers": top_papers,
                "total_papers_fetched": len(papers),
                "agent_trace": state.get("agent_trace", []) + ["RetrievalAgent"],
            }
