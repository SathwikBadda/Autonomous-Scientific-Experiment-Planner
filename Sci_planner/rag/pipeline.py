"""
rag/pipeline.py - RAG pipeline: embed papers → FAISS → similarity search.
Uses local SentenceTransformer model (mixedbread-ai/mxbai-embed-large-v1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model = None
_faiss_index = None
_stored_papers: list[dict] = []


def _load_embedding_model():
    """Lazy-load the local SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    model_path = config["rag"]["embedding_model_path"]
    logger.info("loading_embedding_model", path=model_path)
    try:
        _embedding_model = SentenceTransformer(model_path)
        logger.info("embedding_model_loaded", path=model_path)
    except Exception as e:
        logger.error("embedding_model_load_failed", path=model_path, error=str(e))
        raise RuntimeError(
            f"Failed to load embedding model from '{model_path}'. "
            f"Ensure the model is downloaded at that path. Error: {e}"
        )
    return _embedding_model


def _get_faiss_index(dim: int):
    """Create or return the FAISS index."""
    global _faiss_index
    if _faiss_index is not None:
        return _faiss_index

    import faiss

    index_type = config["rag"].get("faiss_index_type", "FlatIP")
    logger.info("creating_faiss_index", type=index_type, dim=dim)

    if index_type == "FlatIP":
        _faiss_index = faiss.IndexFlatIP(dim)  # cosine similarity (after L2-norm)
    elif index_type == "FlatL2":
        _faiss_index = faiss.IndexFlatL2(dim)
    else:
        _faiss_index = faiss.IndexFlatIP(dim)

    return _faiss_index


def reset_index():
    """Clear the FAISS index and stored papers (call between pipeline runs)."""
    global _faiss_index, _stored_papers
    _faiss_index = None
    _stored_papers = []
    logger.info("faiss_index_reset")


def _build_document_text(paper: dict) -> str:
    """Combine title + abstract for embedding."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    return f"Title: {title}\nAbstract: {abstract}"


def index_papers(papers: list[dict]) -> int:
    """
    Embed papers and store in FAISS.
    Returns the number of papers indexed.
    """
    global _stored_papers

    if not papers:
        logger.warning("no_papers_to_index")
        return 0

    model = _load_embedding_model()
    dim = config["rag"]["embedding_dim"]
    index = _get_faiss_index(dim)

    texts = [_build_document_text(p) for p in papers]
    logger.info("embedding_papers", count=len(texts))

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # required for cosine similarity with FlatIP
        show_progress_bar=False,
        batch_size=16,
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    index.add(embeddings)
    _stored_papers.extend(papers)

    logger.info("papers_indexed", total_in_index=len(_stored_papers))
    return len(papers)


def search_similar_papers(query: str, top_k: int | None = None) -> list[dict]:
    """
    Given a query string, embed it and return top-k similar papers from FAISS.
    """
    if not _stored_papers:
        logger.warning("faiss_index_empty")
        return []

    k = top_k or config["rag"]["top_k"]
    k = min(k, len(_stored_papers))

    model = _load_embedding_model()
    index = _get_faiss_index(config["rag"]["embedding_dim"])

    query_vec = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    query_vec = np.array(query_vec, dtype=np.float32)

    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_stored_papers):
            continue
        paper = dict(_stored_papers[idx])
        paper["similarity_score"] = float(dist)
        results.append(paper)

    logger.info("similarity_search_done", query=query[:80], top_k=k, found=len(results))
    return results


def build_rag_context(papers: list[dict], max_papers: int = 5) -> str:
    """
    Format retrieved papers into a structured string for LLM context.
    """
    context_parts = []
    for i, paper in enumerate(papers[:max_papers], 1):
        score = paper.get("similarity_score", 0)
        context_parts.append(
            f"[Paper {i}] (relevance: {score:.3f})\n"
            f"Title: {paper.get('title', 'N/A')}\n"
            f"ArXiv ID: {paper.get('arxiv_id', 'N/A')}\n"
            f"Published: {paper.get('published', 'N/A')}\n"
            f"Abstract: {paper.get('abstract', 'N/A')[:800]}\n"
        )
    return "\n---\n".join(context_parts)
