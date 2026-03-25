"""
rag/document_loader.py - PDF loading and text chunking logic.
"""
from __future__ import annotations
import fitz  # PyMuPDF
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class PDFLoader:
    """PDF loading and chunking utility."""

    @staticmethod
    def extract_text(file_path: str | Path) -> str:
        """Extract all text from a PDF file."""
        logger.info("extracting_text_from_pdf", path=str(file_path))
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            logger.info("text_extracted_successfully", length=len(text))
        except Exception as e:
            logger.error("pdf_extraction_failed", error=str(e))
            raise RuntimeError(f"Failed to extract text from PDF: {e}")
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        logger.info("chunking_text", size=chunk_size, overlap=chunk_overlap)
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        
        logger.info("chunking_complete", count=len(chunks))
        return chunks
