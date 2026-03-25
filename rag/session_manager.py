"""
rag/session_manager.py - Session-based storage for fetched papers and FAISS indexes.
Each pipeline run creates a unique session folder: sessions/<session_id>/
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"


def _slug(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from text."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len]


def create_session(problem_statement: str = "") -> dict:
    """
    Create a new session directory with the structure:
      sessions/<session_id>/
        papers/    ← raw JSON + markdown per paper
        index/     ← FAISS index files
        metadata.json
    Returns a session dict with session_id and all folder paths.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(problem_statement)[:40] if problem_statement else "run"
    session_id = f"{ts}_{slug}"

    session_dir = SESSIONS_ROOT / session_id
    papers_dir = session_dir / "papers"
    index_dir = session_dir / "index"

    papers_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "papers_dir": str(papers_dir),
        "index_dir": str(index_dir),
        "created_at": datetime.now().isoformat(),
        "problem_statement": problem_statement,
    }

    # Write session metadata
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(session, f, indent=2)

    logger.info("session_created", session_id=session_id, path=str(session_dir))
    return session


def save_paper(session: dict, paper: dict) -> str:
    """
    Save a paper as both .json and .md files.
    Returns the path to the markdown file.
    """
    papers_dir = Path(session["papers_dir"])
    title = paper.get("title", "untitled")
    slug = _slug(title)
    if not slug:
        slug = f"paper_{hash(title) & 0xFFFF}"

    # Save JSON
    json_path = papers_dir / f"{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=2, ensure_ascii=False)

    # Convert to Markdown
    md_content = paper_to_markdown(paper)
    md_path = papers_dir / f"{slug}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return str(md_path)


def save_all_papers(session: dict, papers: list) -> list:
    """Save all papers and return list of markdown paths."""
    md_paths = []
    for paper in papers:
        try:
            path = save_paper(session, paper)
            md_paths.append(path)
        except Exception as e:
            logger.warning("paper_save_failed", title=paper.get("title", "?"), error=str(e))
    logger.info("papers_saved", count=len(md_paths), dir=session["papers_dir"])
    return md_paths


def paper_to_markdown(paper: dict) -> str:
    """
    Convert a paper dict into a structured Markdown document with subheaders.
    Each section is suitable for chunking individually.
    """
    lines = []
    title = paper.get("title", "Untitled Paper").strip()
    lines.append(f"# {title}\n")

    # Metadata block
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", "")
    arxiv_id = paper.get("arxiv_id", "")
    url = paper.get("url", paper.get("pdf_url", ""))
    venue = paper.get("venue", paper.get("journal", ""))

    lines.append("## Metadata\n")
    if authors:
        lines.append(f"- **Authors:** {authors}")
    if year:
        lines.append(f"- **Year:** {year}")
    if venue:
        lines.append(f"- **Venue:** {venue}")
    if arxiv_id:
        lines.append(f"- **arXiv ID:** {arxiv_id}")
    if url:
        lines.append(f"- **URL:** {url}")
    lines.append("")

    # Abstract
    abstract = paper.get("abstract", "").strip()
    if abstract:
        lines.append("## Abstract\n")
        lines.append(abstract)
        lines.append("")

    # Methods / techniques (often included in structured papers)
    methods = paper.get("methods", paper.get("methodology", "")).strip()
    if methods:
        lines.append("## Methods\n")
        lines.append(methods)
        lines.append("")

    # Results / evaluation
    results = paper.get("results", paper.get("evaluation", "")).strip()
    if results:
        lines.append("## Results\n")
        lines.append(results)
        lines.append("")

    # Limitations
    limitations = paper.get("limitations", "").strip()
    if limitations:
        lines.append("## Limitations\n")
        lines.append(limitations)
        lines.append("")

    # Contributions
    contributions = paper.get("contributions", "").strip()
    if contributions:
        lines.append("## Key Contributions\n")
        lines.append(contributions)
        lines.append("")

    # Categories / topics
    categories = paper.get("categories", [])
    if isinstance(categories, list) and categories:
        lines.append("## Topics\n")
        lines.append(", ".join(categories))
        lines.append("")

    return "\n".join(lines)


def get_session_summary(session: dict) -> dict:
    """Return summary info about files created in this session."""
    papers_dir = Path(session["papers_dir"])
    index_dir = Path(session["index_dir"])

    json_files = list(papers_dir.glob("*.json"))
    md_files = list(papers_dir.glob("*.md"))
    index_files = list(index_dir.glob("*"))

    return {
        "session_id": session["session_id"],
        "papers_saved": len(json_files),
        "markdown_files": len(md_files),
        "index_files": [f.name for f in index_files],
        "session_dir": session["session_dir"],
    }
