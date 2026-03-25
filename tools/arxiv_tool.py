"""
tools/arxiv_tool.py - MCP-style arXiv search tool.
Fetches paper metadata via arXiv API, then downloads full paper text from ar5iv.org (HTML).
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

import arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ar5iv provides full HTML of arXiv papers (latex compiled to HTML)
AR5IV_BASE = "https://ar5iv.org/html"
SEMANTIC_BASE = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"


# --- MCP Tool Schema ---
ARXIV_TOOL_SCHEMA: dict = {
    "name": "search_arxiv",
    "description": (
        "Search the arXiv preprint server for scientific papers. "
        "Returns papers with full text sections (abstract, methods, results, limitations). "
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
                "description": "Number of papers to retrieve (1-10).",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
}


def _clean_text(text: str) -> str:
    """Clean extracted HTML text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[\d,\s]+\]", "", text)   # remove citation brackets
    return text.strip()


def _extract_section(soup, section_names: list) -> str:
    """Extract text from named sections in BeautifulSoup HTML."""
    for name in section_names:
        # Try finding section by heading text
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            if name.lower() in tag.get_text().lower():
                texts = []
                for sib in tag.find_next_siblings():
                    if sib.name in ["h1", "h2", "h3", "h4"]:
                        break
                    texts.append(sib.get_text())
                content = _clean_text(" ".join(texts))
                if len(content) > 100:
                    return content[:3000]
    return ""


def _fetch_full_paper_ar5iv(arxiv_id: str) -> dict:
    """
    Fetch the full paper from ar5iv.org (HTML format).
    Returns a dict of section_name → text.
    """
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    url = f"{AR5IV_BASE}/{clean_id}"

    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "SciPlanner/1.0"})
        if resp.status_code != 200:
            logger.warning("ar5iv_fetch_failed", arxiv_id=arxiv_id, status=resp.status_code)
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")

        sections = {
            "introduction": _extract_section(soup, ["introduction", "background", "motivation"]),
            "methodology": _extract_section(soup, ["method", "methodology", "approach", "model", "architecture"]),
            "results": _extract_section(soup, ["result", "evaluation", "experiment", "performance"]),
            "conclusion": _extract_section(soup, ["conclusion", "discussion", "summary", "future work"]),
            "limitations": _extract_section(soup, ["limitation", "weakness", "constraint", "failure"]),
        }

        # Also get evaluation metrics table if present
        metrics_text = ""
        for table in soup.find_all("table")[:5]:
            ttext = table.get_text()
            if any(m in ttext.lower() for m in ["accuracy", "f1", "bleu", "rouge", "precision", "recall"]):
                metrics_text = _clean_text(ttext)[:1500]
                break
        if metrics_text:
            sections["evaluation_metrics"] = metrics_text

        non_empty = {k: v for k, v in sections.items() if v}
        logger.info("ar5iv_paper_fetched", arxiv_id=arxiv_id, sections=list(non_empty.keys()))
        return non_empty

    except Exception as e:
        logger.warning("ar5iv_fetch_error", arxiv_id=arxiv_id, error=str(e))
        return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_arxiv(query: str, max_results: int) -> list:
    """Internal arxiv fetch + ar5iv full paper download."""
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    for paper in client.results(search):
        arxiv_id = paper.entry_id.split("/")[-1]

        # Base metadata from arXiv API
        entry = {
            "title": paper.title,
            "abstract": paper.summary,         # full abstract
            "arxiv_id": arxiv_id,
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "authors": [a.name for a in paper.authors[:5]],
            "published": str(paper.published.date()) if paper.published else None,
            "year": str(paper.published.year) if paper.published else None,
            "categories": paper.categories,
            "query_used": query,
        }

        # Fetch full sections from ar5iv
        full_sections = _fetch_full_paper_ar5iv(arxiv_id)
        entry.update(full_sections)   # adds introduction, methodology, results, conclusion, limitations

        results.append(entry)
        time.sleep(0.5)  # be polite to ar5iv

    return results


def execute_search_arxiv(tool_input: dict) -> dict:
    """Execute the search_arxiv MCP tool call."""
    query = tool_input.get("query", "")
    max_results = min(int(tool_input.get("max_results", config["arxiv"]["max_results"])), 10)

    logger.info("arxiv_search_start", query=query, max_results=max_results)
    if not query.strip():
        return {"error": "Empty query", "papers": []}

    try:
        papers = _fetch_arxiv(query, max_results)
        logger.info("arxiv_search_done", query=query, found=len(papers))
        return {"papers": papers, "total_retrieved": len(papers), "query": query}
    except Exception as e:
        logger.error("arxiv_search_failed", query=query, error=str(e))
        return {"error": str(e), "papers": [], "total_retrieved": 0, "query": query}


# Tool registry (shared with dataset_tool)
TOOL_REGISTRY: dict = {
    "search_arxiv": {
        "schema": ARXIV_TOOL_SCHEMA,
        "executor": execute_search_arxiv,
    }
}


def register_tool(name: str, schema: dict, executor) -> None:
    """Register an additional MCP tool dynamically."""
    TOOL_REGISTRY[name] = {"schema": schema, "executor": executor}


def _register_extra_tools():
    """Register all extra tools (dataset search, etc.) into the unified registry."""
    try:
        from tools.dataset_tool import DATASET_SEARCH_TOOL_SCHEMA, execute_search_datasets
        register_tool("search_datasets", DATASET_SEARCH_TOOL_SCHEMA, execute_search_datasets)
        logger.info("dataset_tool_registered")
    except Exception as e:
        logger.warning("dataset_tool_registration_failed", error=str(e))


# Auto-register all tools at import time
_register_extra_tools()


def get_all_tool_schemas() -> list:
    return [v["schema"] for v in TOOL_REGISTRY.values()]


def dispatch_tool_call(tool_name: str, tool_input: dict) -> Any:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]["executor"](tool_input)
