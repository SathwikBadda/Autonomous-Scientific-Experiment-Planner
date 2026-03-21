"""
tools/arxiv_tool.py - MCP-style arXiv search tool.
The LLM calls this tool dynamically; no hardcoded retrieval logic.
"""
from __future__ import annotations

import time
from typing import Any

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

# --- MCP Tool Definition (passed to Claude as a tool) ---
ARXIV_TOOL_SCHEMA: dict[str, Any] = {
    "name": "search_arxiv",
    "description": (
        "Search the arXiv preprint server for scientific papers. "
        "Returns a list of papers with title, abstract, arxiv_id, and URL. "
        "Use this to retrieve relevant literature for any research query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string. Use specific technical terms.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to retrieve (1–20).",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _fetch_arxiv(query: str, max_results: int) -> list[dict]:
    """Internal arxiv fetch with retry logic."""
    arxiv_cfg = config.get("arxiv", {})
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                "abstract": paper.summary[:1500],  # truncate for LLM context
                "arxiv_id": paper.entry_id.split("/")[-1],
                "url": paper.entry_id,
                "authors": [a.name for a in paper.authors[:3]],
                "published": str(paper.published.date()) if paper.published else None,
                "categories": paper.categories,
                "query_used": query,
            }
        )
    return results


def execute_search_arxiv(tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the search_arxiv MCP tool call.
    This is invoked when Claude returns a tool_use block for 'search_arxiv'.
    """
    query = tool_input.get("query", "")
    max_results = int(tool_input.get("max_results", config["arxiv"]["max_results"]))
    max_results = min(max_results, 20)

    logger.info("arxiv_search_start", query=query, max_results=max_results)

    if not query.strip():
        return {"error": "Empty search query provided", "papers": []}

    try:
        papers = _fetch_arxiv(query, max_results)
        logger.info("arxiv_search_done", query=query, found=len(papers))
        return {
            "papers": papers,
            "total_retrieved": len(papers),
            "query": query,
        }
    except Exception as e:
        logger.error("arxiv_search_failed", query=query, error=str(e))
        return {
            "error": str(e),
            "papers": [],
            "total_retrieved": 0,
            "query": query,
        }


# Registry of all available MCP tools
TOOL_REGISTRY: dict[str, Any] = {
    "search_arxiv": {
        "schema": ARXIV_TOOL_SCHEMA,
        "executor": execute_search_arxiv,
    }
}


def get_all_tool_schemas() -> list[dict]:
    """Return all tool schemas for passing to the LLM."""
    return [v["schema"] for v in TOOL_REGISTRY.values()]


def dispatch_tool_call(tool_name: str, tool_input: dict) -> Any:
    """Route a tool_use block from Claude to the correct executor."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]["executor"](tool_input)
