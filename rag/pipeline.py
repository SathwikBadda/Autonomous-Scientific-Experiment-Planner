"""
rag/pipeline.py - RAG pipeline using ChromaDB (replaces FAISS).
ChromaDB uses HNSWLIB under the hood — stable on Apple Silicon M1/M2.
Per-session persistent collections stored on disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level state
_embedding_model = None
_chroma_client = None
_papers_collection = None   # for fetched arXiv paper chunks
_pdf_collection = None      # for uploaded PDF chunks

_stored_paper_chunks: list[dict] = []   # mirrors what's in ChromaDB for fast fallback
_stored_pdf_chunks: list[str] = []


def _load_embedding_model():
    """Lazy-load the SentenceTransformer model (CPU only — avoids MPS segfault)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer
    model_path = config["rag"]["embedding_model_path"]
    logger.info("loading_embedding_model", path=model_path, device="cpu")
    try:
        _embedding_model = SentenceTransformer(model_path, device="cpu")
        logger.info("embedding_model_loaded", path=model_path, device="cpu")
    except Exception as e:
        logger.error("embedding_model_load_failed", error=str(e))
        raise RuntimeError(f"Failed to load embedding model: {e}")
    return _embedding_model


def _get_chroma_client(persist_dir: Optional[str] = None):
    """Get or create a ChromaDB client (in-memory or persistent)."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    import chromadb
    if persist_dir:
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        logger.info("chromadb_persistent_client", path=persist_dir)
    else:
        _chroma_client = chromadb.Client()
        logger.info("chromadb_ephemeral_client")
    return _chroma_client


def _get_papers_collection(session_index_dir: Optional[str] = None):
    """Get or create the papers ChromaDB collection."""
    global _papers_collection
    if _papers_collection is not None:
        return _papers_collection
    client = _get_chroma_client(persist_dir=session_index_dir)
    _papers_collection = client.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"},
    )
    return _papers_collection


def _get_pdf_collection(session_index_dir: Optional[str] = None):
    """Get or create the PDF chunks ChromaDB collection."""
    global _pdf_collection
    if _pdf_collection is not None:
        return _pdf_collection
    client = _get_chroma_client(persist_dir=session_index_dir)
    _pdf_collection = client.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    return _pdf_collection


def reset_index():
    """Reset all collections and stored data for a fresh run."""
    global _chroma_client, _papers_collection, _pdf_collection
    global _stored_paper_chunks, _stored_pdf_chunks
    _chroma_client = None
    _papers_collection = None
    _pdf_collection = None
    _stored_paper_chunks = []
    _stored_pdf_chunks = []
    logger.info("chroma_indexes_reset")


def chunk_paper_sections(paper: dict, chunk_size: int = 400, overlap: int = 60) -> list[dict]:
    """
    Chunk a paper dict into section-level pieces.
    Returns list of {chunk_text, title, section, source_paper}.
    """
    title = paper.get("title", "Untitled")
    sections = {
        "Abstract": paper.get("abstract", ""),
        "Introduction": paper.get("introduction", ""),
        "Methods": paper.get("methodology", paper.get("methods", "")),
        "Results": paper.get("results", paper.get("evaluation", "")),
        "Conclusion": paper.get("conclusion", ""),
        "Limitations": paper.get("limitations", ""),
        "EvaluationMetrics": paper.get("evaluation_metrics", ""),
    }

    chunks = []
    for section_name, text in sections.items():
        if not text or not text.strip():
            continue
        words = text.split()
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start: start + chunk_size]
            if len(chunk_words) < 15:
                continue
            chunk_text = f"[{title}] [{section_name}]\n" + " ".join(chunk_words)
            chunks.append({
                "chunk_text": chunk_text,
                "title": title,
                "section": section_name,
                "source_paper": paper,
            })

    # Fallback to abstract only
    if not chunks:
        abstract = paper.get("abstract", "")
        if abstract:
            chunks.append({
                "chunk_text": f"[{title}] [Abstract]\n{abstract}",
                "title": title,
                "section": "Abstract",
                "source_paper": paper,
            })
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts and return as list of float lists for ChromaDB."""
    model = _load_embedding_model()
    import numpy as np
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8)
    return embeddings.tolist()


def index_papers(papers: list[dict], session_index_dir: Optional[str] = None) -> int:
    """Chunk papers by section, embed, and store in ChromaDB papers collection."""
    global _stored_paper_chunks
    if not papers:
        return 0

    collection = _get_papers_collection(session_index_dir)

    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper_sections(paper))

    if not all_chunks:
        return 0

    texts = [c["chunk_text"] for c in all_chunks]
    logger.info("embedding_paper_chunks_chroma", count=len(texts))

    # Batch in groups of 50 to avoid memory spikes
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        batch_texts = [c["chunk_text"] for c in batch]
        batch_ids = [f"paper_{i + j}_{hash(c['chunk_text']) & 0xFFFFFF}" for j, c in enumerate(batch)]
        batch_metas = [
            {
                "title": c["title"][:200],
                "section": c["section"],
                "paper_idx": i + j,
            }
            for j, c in enumerate(batch)
        ]
        batch_embeddings = _embed(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )

    _stored_paper_chunks.extend(all_chunks)
    logger.info("paper_chunks_indexed_chroma", total=len(_stored_paper_chunks))
    return len(all_chunks)


def index_chunks(chunks: list[str], session_index_dir: Optional[str] = None) -> int:
    """Embed raw text chunks (from uploaded PDFs) and store in ChromaDB."""
    global _stored_pdf_chunks
    if not chunks:
        return 0

    collection = _get_pdf_collection(session_index_dir)
    logger.info("embedding_pdf_chunks_chroma", count=len(chunks))

    for i in range(0, len(chunks), 50):
        batch = chunks[i: i + 50]
        ids = [f"pdf_{i + j}_{hash(c) & 0xFFFFFF}" for j, c in enumerate(batch)]
        embeddings = _embed(batch)
        collection.add(ids=ids, embeddings=embeddings, documents=batch)

    _stored_pdf_chunks.extend(chunks)
    return len(chunks)


def search_similar_papers(query: str, top_k: Optional[int] = None) -> list:
    """
    Search the paper chunks collection for the most relevant papers.
    Returns deduplicated paper dicts sorted by relevance.
    """
    if not _stored_paper_chunks or _papers_collection is None:
        return []

    k = min(top_k or config["rag"]["top_k"] * 5, len(_stored_paper_chunks))
    q_emb = _embed([query])

    results = _papers_collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    seen_titles: set = set()
    papers = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        title = meta.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            # Find full source paper from stored chunks
            paper_idx = meta.get("paper_idx", 0)
            if paper_idx < len(_stored_paper_chunks):
                paper = dict(_stored_paper_chunks[paper_idx]["source_paper"])
            else:
                paper = {"title": title}
            paper["similarity_score"] = float(1 - dist)   # cosine: distance → similarity
            papers.append(paper)

    top_k_final = top_k or config["rag"]["top_k"]
    return papers[:top_k_final]


def search_by_section(query: str, section: str, top_k: int = 5) -> list:
    """
    Retrieve chunks from a specific section type (e.g., 'Limitations', 'Methods').
    Uses ChromaDB where filter.
    """
    if not _stored_paper_chunks or _papers_collection is None:
        return []

    q_emb = _embed([query])
    k = min(top_k * 3, len(_stored_paper_chunks))

    try:
        results = _papers_collection.query(
            query_embeddings=q_emb,
            n_results=k,
            where={"section": section},
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        return docs[:top_k]
    except Exception as e:
        logger.warning("section_search_failed", section=section, error=str(e))
        # Fallback: filter from stored chunks manually
        matching = [
            c["chunk_text"]
            for c in _stored_paper_chunks
            if c.get("section", "").lower() == section.lower()
        ]
        return matching[:top_k]


def search_pdf_chunks(query: str, top_k: Optional[int] = None) -> list:
    """Retrieve top-k relevant chunks from PDF ChromaDB collection."""
    if not _stored_pdf_chunks or _pdf_collection is None:
        return []

    k = min(top_k or config["rag"]["top_k"], len(_stored_pdf_chunks))
    q_emb = _embed([query])

    results = _pdf_collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents"],
    )
    return results["documents"][0]


def save_index_to_disk(index_dir: str) -> dict:
    """
    ChromaDB PersistentClient already writes to disk automatically.
    This function just records what was saved and returns summary.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Write a summary JSON for reference
    summary = {
        "paper_chunks": len(_stored_paper_chunks),
        "pdf_chunks": len(_stored_pdf_chunks),
        "unique_papers": len({c["title"] for c in _stored_paper_chunks}),
        "index_dir": str(index_dir),
        "backend": "ChromaDB",
    }
    with open(index_dir / "index_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("chroma_index_summary_saved", **summary)
    return summary


def build_rag_context(papers: list = None, chunks: list = None, max_items: int = 5) -> str:
    """Format retrieved papers and chunks into a rich context string for LLM agents."""
    context_parts = []

    if chunks:
        context_parts.append("### CONTEXT FROM UPLOADED PDF")
        for i, chunk in enumerate(chunks[:max_items], 1):
            context_parts.append(f"[PDF Chunk {i}]\n{chunk}\n")

    if papers:
        context_parts.append("### RETRIEVED LITERATURE (arXiv)")
        for i, paper in enumerate(papers[:max_items], 1):
            score = paper.get("similarity_score", 0)
            context_parts.append(
                f"[Paper {i}] (relevance: {score:.3f})\n"
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {', '.join(paper.get('authors', []))}\n"
                f"Year: {paper.get('year', 'N/A')}\n"
                f"Abstract: {paper.get('abstract', 'N/A')[:800]}\n"
                f"Methods: {paper.get('methodology', paper.get('methods', 'Not extracted'))[:400]}\n"
                f"Results: {paper.get('results', 'Not extracted')[:400]}\n"
                f"Limitations: {paper.get('limitations', 'Not explicitly stated')[:400]}\n"
            )

    return "\n\n---\n\n".join(context_parts)
