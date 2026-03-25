"""
tools/dataset_tool.py - MCP-style dataset discovery tool.
Searches HuggingFace Hub and Kaggle for relevant research datasets.
"""
from __future__ import annotations

import requests
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

HF_DATASETS_API = "https://huggingface.co/api/datasets"
KAGGLE_SEARCH_URL = "https://www.kaggle.com/api/v1/datasets/list"


DATASET_SEARCH_TOOL_SCHEMA: dict = {
    "name": "search_datasets",
    "description": (
        "Search for publicly available research datasets on HuggingFace Hub. "
        "Returns dataset names, descriptions, download counts, and direct URLs. "
        "Use this to find real datasets relevant to your research problem."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for datasets (e.g. 'question answering', 'hallucination detection')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of datasets to return (1-10).",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
            "task_category": {
                "type": "string",
                "description": "Optional HuggingFace task category filter (e.g., 'text-classification', 'question-answering', 'summarization')",
            },
        },
        "required": ["query"],
    },
}


def _search_huggingface(query: str, max_results: int, task_category: str = "") -> list:
    """Search HuggingFace Hub datasets API."""
    params = {
        "search": query,
        "limit": max_results,
        "sort": "downloads",
        "direction": -1,
    }
    if task_category:
        params["filter"] = task_category

    try:
        resp = requests.get(
            HF_DATASETS_API,
            params=params,
            timeout=10,
            headers={"User-Agent": "SciPlanner/1.0"},
        )
        if resp.status_code != 200:
            logger.warning("hf_api_failed", status=resp.status_code)
            return []

        datasets = []
        for ds in resp.json():
            ds_id = ds.get("id", "")
            name = ds.get("name", ds_id.split("/")[-1] if "/" in ds_id else ds_id)
            description = ds.get("description", "No description available")
            if isinstance(description, str):
                description = description[:300]

            datasets.append({
                "name": name,
                "id": ds_id,
                "url": f"https://huggingface.co/datasets/{ds_id}",
                "description": description,
                "downloads": ds.get("downloads", 0),
                "tags": ds.get("tags", [])[:8],
                "source": "HuggingFace",
                "license": ds.get("license", "Unknown"),
            })

        logger.info("hf_datasets_found", query=query, count=len(datasets))
        return datasets

    except Exception as e:
        logger.warning("hf_search_error", query=query, error=str(e))
        return []


def _get_fallback_datasets(query: str) -> list:
    """
    Return curated well-known datasets if HF API returns nothing relevant.
    These are commonly used in NLP/ML research.
    """
    keyword_map = {
        "hallucination": [
            {"name": "TruthfulQA", "url": "https://huggingface.co/datasets/truthful_qa",
             "description": "Questions where LLMs tend to hallucinate", "source": "HuggingFace", "downloads": 50000},
            {"name": "HaluEval", "url": "https://huggingface.co/datasets/pminervini/HaluEval",
             "description": "Hallucination evaluation benchmark for LLMs", "source": "HuggingFace", "downloads": 20000},
        ],
        "question answering": [
            {"name": "SQuAD 2.0", "url": "https://huggingface.co/datasets/rajpurkar/squad_v2",
             "description": "Reading comprehension with unanswerable questions", "source": "HuggingFace", "downloads": 300000},
            {"name": "Natural Questions", "url": "https://huggingface.co/datasets/google-research-datasets/natural_questions",
             "description": "Real Google search questions with Wikipedia answers", "source": "HuggingFace", "downloads": 150000},
        ],
        "summarization": [
            {"name": "CNN/DailyMail", "url": "https://huggingface.co/datasets/cnn_dailymail",
             "description": "News articles with highlights/summaries", "source": "HuggingFace", "downloads": 200000},
            {"name": "XSum", "url": "https://huggingface.co/datasets/EdinburghNLP/xsum",
             "description": "Extreme summarization of BBC news articles", "source": "HuggingFace", "downloads": 100000},
        ],
        "retrieval": [
            {"name": "BEIR", "url": "https://huggingface.co/datasets/BeIR/beir",
             "description": "Heterogeneous IR benchmark", "source": "HuggingFace", "downloads": 80000},
            {"name": "MS MARCO", "url": "https://huggingface.co/datasets/ms_marco",
             "description": "Large-scale passage retrieval dataset", "source": "HuggingFace", "downloads": 250000},
        ],
    }

    query_lower = query.lower()
    results = []
    for keyword, datasets in keyword_map.items():
        if keyword in query_lower:
            results.extend(datasets)
    return results[:5]


def execute_search_datasets(tool_input: dict) -> dict:
    """Execute the search_datasets MCP tool call."""
    query = tool_input.get("query", "")
    max_results = min(int(tool_input.get("max_results", 5)), 10)
    task_category = tool_input.get("task_category", "")

    logger.info("dataset_search_start", query=query, max_results=max_results)

    if not query.strip():
        return {"error": "Empty query", "datasets": []}

    # Primary: HuggingFace
    datasets = _search_huggingface(query, max_results, task_category)

    # Fallback: curated list
    if len(datasets) < 2:
        fallback = _get_fallback_datasets(query)
        datasets = datasets + fallback

    datasets = datasets[:max_results]

    logger.info("dataset_search_done", query=query, found=len(datasets))
    return {
        "datasets": datasets,
        "total_found": len(datasets),
        "query": query,
        "sources": ["HuggingFace Hub"],
    }
