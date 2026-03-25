"""
rag/paper_downloader.py - Downloads full arXiv PDFs, extracts text with PyMuPDF,
converts to structured Markdown with subheadings, and indexes in ChromaDB.

Pipeline:
  arXiv ID → download PDF → extract text via pymupdf → detect sections →
  save .pdf + .md → chunk by section → embed with all-MiniLM-L6-v2 → ChromaDB
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# Sections to detect (ordered by typical paper structure)
SECTION_PATTERNS = [
    ("Abstract", r"abstract"),
    ("Introduction", r"introduction|background|motivation"),
    ("Related Work", r"related\s+work|prior\s+work|literature\s+review"),
    ("Methodology", r"method(?:ology)?|approach|framework|model|architecture|proposed"),
    ("Experiments", r"experiment(?:al\s+setup)?|implementation\s+details|setup"),
    ("Results", r"result|evaluation|performance|analysis|comparison"),
    ("Discussion", r"discussion|analysis|ablation"),
    ("Conclusion", r"conclusion|summary|future\s+work|conclud"),
    ("Limitations", r"limitation|weakness|failure|constraint"),
    ("References", r"reference|bibliography"),
]


def download_pdf(arxiv_id: str, output_dir: str) -> Optional[str]:
    """
    Download the PDF for a given arXiv ID.
    Returns path to downloaded PDF or None on failure.
    """
    clean_id = re.sub(r"v\d+$", "", arxiv_id)  # remove version suffix
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    output_path = Path(output_dir) / f"{clean_id}.pdf"

    if output_path.exists():
        logger.info("pdf_already_exists", arxiv_id=arxiv_id, path=str(output_path))
        return str(output_path)

    logger.info("downloading_pdf", arxiv_id=arxiv_id, url=pdf_url)
    try:
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "SciPlanner/1.0 (research tool)"})
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
            with open(output_path, "wb") as f:
                f.write(resp.content)
            size_kb = output_path.stat().st_size // 1024
            logger.info("pdf_downloaded", arxiv_id=arxiv_id, size_kb=size_kb, path=str(output_path))
            return str(output_path)
        else:
            logger.warning("pdf_download_failed", arxiv_id=arxiv_id, status=resp.status_code)
            return None
    except Exception as e:
        logger.error("pdf_download_error", arxiv_id=arxiv_id, error=str(e))
        return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from PDF using PyMuPDF."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            pages_text.append(text)
        doc.close()
        full_text = "\n".join(pages_text)
        logger.info("pdf_text_extracted", path=pdf_path, chars=len(full_text))
        return full_text
    except Exception as e:
        logger.error("pdf_extraction_failed", path=pdf_path, error=str(e))
        return ""


def detect_sections(text: str) -> dict:
    """
    Detect major sections in raw PDF text using regex patterns.
    Returns dict: {section_name → section_text}
    """
    lines = text.split("\n")
    sections: dict = {}
    current_section = "Preamble"
    current_lines: list = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        # Check if this line is a section header
        matched_section = None
        for section_name, pattern in SECTION_PATTERNS:
            # Match headers: short lines that contain the section keyword
            if (len(stripped) < 80 and
                    re.search(pattern, stripped, re.IGNORECASE) and
                    re.match(r"^[\d\s\.\u2013\-]*[A-Z]", stripped)):
                matched_section = section_name
                break

        if matched_section and matched_section != current_section:
            # Save previous section
            section_text = _clean_section_text("\n".join(current_lines))
            if section_text and len(section_text) > 100:
                sections[current_section] = section_text
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        section_text = _clean_section_text("\n".join(current_lines))
        if section_text and len(section_text) > 100:
            sections[current_section] = section_text

    # If no sections detected, fall back to splitting by paragraph groups
    if len(sections) <= 1:
        sections = _fallback_split(text)

    logger.info("sections_detected", sections=list(sections.keys()))
    return sections


def _clean_section_text(text: str) -> str:
    """Remove excessive whitespace and common PDF artifacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse multiple blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)         # collapse multiple spaces
    text = re.sub(r"- \n", "", text)                # join hyphenated line breaks
    text = re.sub(r"\x0c", "\n", text)              # form feeds
    return text.strip()


def _fallback_split(text: str) -> dict:
    """Fallback: split text into chunks of ~2000 words if section detection fails."""
    words = text.split()
    chunk_size = 2000
    sections = {}
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        sections[f"Section_{i // chunk_size + 1}"] = chunk
    return sections


def pdf_to_markdown(title: str, sections: dict, metadata: dict = None) -> str:
    """
    Convert detected sections to a structured Markdown document.
    """
    lines = [f"# {title}\n"]

    if metadata:
        lines.append("## Metadata\n")
        for k, v in metadata.items():
            if v:
                lines.append(f"- **{k.title().replace('_', ' ')}:** {v}")
        lines.append("")

    # Write sections in order
    section_order = [s[0] for s in SECTION_PATTERNS] + ["Preamble"]
    written = set()

    for section_name in section_order:
        if section_name in sections and section_name not in written:
            lines.append(f"## {section_name}\n")
            lines.append(sections[section_name])
            lines.append("")
            written.add(section_name)

    # Write any remaining sections not in the order list
    for section_name, text in sections.items():
        if section_name not in written:
            lines.append(f"## {section_name}\n")
            lines.append(text)
            lines.append("")

    return "\n".join(lines)


def chunk_section_text(section_name: str, text: str,
                        chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Sliding window chunking of section text, prefixed with section name."""
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start: start + chunk_size]
        if len(chunk_words) < 20:
            continue
        chunk_text = f"[{section_name}]\n" + " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks


def process_paper_to_markdown_and_chunks(
    paper: dict,
    output_dir: str,
) -> dict:
    """
    Full pipeline for one paper:
    1. Download PDF from arXiv
    2. Extract text with pymupdf
    3. Detect sections
    4. Save as structured .md
    5. Return chunked sections for embedding

    Returns: {
        pdf_path, md_path, sections, chunks,
        success: bool
    }
    """
    arxiv_id = paper.get("arxiv_id", "")
    title = paper.get("title", "Untitled")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "arxiv_id": arxiv_id,
        "title": title,
        "pdf_path": None,
        "md_path": None,
        "sections": {},
        "chunks": [],
        "success": False,
    }

    if not arxiv_id:
        logger.warning("no_arxiv_id", title=title)
        return result

    # 1. Download PDF
    pdf_path = download_pdf(arxiv_id, str(output_dir))
    result["pdf_path"] = pdf_path

    if not pdf_path:
        logger.warning("pdf_download_skipped_using_abstract", arxiv_id=arxiv_id)
        # Fallback: use existing metadata
        sections = {}
        for field in ["abstract", "introduction", "methodology", "results", "conclusion", "limitations"]:
            val = paper.get(field, "")
            if val and len(val) > 50:
                sections[field.capitalize()] = val
        result["sections"] = sections
    else:
        # 2. Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            return result

        # 3. Detect sections
        sections = detect_sections(raw_text)
        # Augment with abstract from API (usually cleaner)
        if paper.get("abstract"):
            sections["Abstract"] = paper["abstract"]
        result["sections"] = sections

    # 4. Save as structured Markdown
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_]+", "_", slug)[:60]
    md_content = pdf_to_markdown(
        title=title,
        sections=result["sections"],
        metadata={
            "authors": ", ".join(paper.get("authors", [])),
            "year": paper.get("year", ""),
            "arxiv_id": arxiv_id,
            "url": paper.get("url", ""),
        },
    )
    md_path = output_dir / f"{slug}_full.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    result["md_path"] = str(md_path)
    logger.info("markdown_saved", path=str(md_path), sections=list(result["sections"].keys()))

    # 5. Create chunks per section
    all_chunks = []
    for section_name, text in result["sections"].items():
        section_chunks = chunk_section_text(section_name, text)
        for chunk in section_chunks:
            all_chunks.append({
                "chunk_text": f"[{title}] {chunk}",
                "title": title,
                "section": section_name,
                "arxiv_id": arxiv_id,
            })

    result["chunks"] = all_chunks
    result["success"] = True
    logger.info("paper_processed", arxiv_id=arxiv_id, chunks=len(all_chunks), sections=len(result["sections"]))
    return result
