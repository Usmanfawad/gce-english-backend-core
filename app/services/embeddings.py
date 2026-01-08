"""Embedding generation and chunking utilities."""

from __future__ import annotations

import re
from typing import List, Optional, Dict, Any

from loguru import logger
from openai import OpenAI

from app.config.settings import settings
from app.db.supabase import PaperChunk


class EmbeddingError(RuntimeError):
    """Raised when embedding operations fail."""


def get_openai_client() -> OpenAI:
    """Get the OpenAI client instance."""
    if not settings.openai_api_key:
        raise EmbeddingError(
            "OpenAI API key not configured. Set OPENAI_API_KEY."
        )
    return OpenAI(api_key=settings.openai_api_key)


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from a paper filename.
    
    Supported formats:
    - GCE Official: 2016_GCE-O-LEVEL-ENGLISH-1128-Paper-1.pdf
    - School papers: Sec4_English_2021_SA2_admiralty_Paper1.pdf
    - With timestamps: 2015_GCE-O-LEVEL-ENGLISH-1128-Paper-2-20251107-164330.txt
    
    Returns dict with year, paper_type, school, etc.
    """
    metadata: Dict[str, Any] = {
        "year": None,
        "paper_type": None,
        "exam_code": None,
        "school": None,
        "is_answer_sheet": False,
        "raw_filename": filename,
    }
    
    filename_lower = filename.lower()
    
    # Check if this is an answer sheet (skip these)
    if "_ans" in filename_lower or "_answer" in filename_lower:
        metadata["is_answer_sheet"] = True
        return metadata
    
    # Extract year - multiple patterns
    # Pattern 1: Year at start (2016_GCE...)
    year_match = re.match(r"^(\d{4})", filename)
    if year_match:
        metadata["year"] = year_match.group(1)
    else:
        # Pattern 2: Year in middle (Sec4_English_2021_SA2...)
        year_match = re.search(r"_(\d{4})_", filename)
        if year_match:
            metadata["year"] = year_match.group(1)
    
    # Extract paper type - multiple patterns
    # Pattern 1: Paper-1, Paper-2 (with hyphen)
    paper_match = re.search(r"Paper-(\d)", filename, re.IGNORECASE)
    if paper_match:
        paper_num = paper_match.group(1)
        metadata["paper_type"] = f"paper_{paper_num}"
    else:
        # Pattern 2: Paper1, Paper2 (without hyphen) - for school papers
        paper_match = re.search(r"Paper(\d)", filename, re.IGNORECASE)
        if paper_match:
            paper_num = paper_match.group(1)
            metadata["paper_type"] = f"paper_{paper_num}"
        else:
            # Pattern 3: _P1, _P2 shorthand
            paper_match = re.search(r"_P(\d)[\._]", filename, re.IGNORECASE)
            if paper_match:
                paper_num = paper_match.group(1)
                metadata["paper_type"] = f"paper_{paper_num}"
    
    # Extract exam code (e.g., 1128)
    code_match = re.search(r"ENGLISH-(\d+)", filename, re.IGNORECASE)
    if code_match:
        metadata["exam_code"] = code_match.group(1)
    
    # Extract school name for school papers (Sec4_English_2021_SA2_schoolname_Paper1.pdf)
    school_match = re.search(r"SA\d_([a-zA-Z]+)_Paper", filename, re.IGNORECASE)
    if school_match:
        metadata["school"] = school_match.group(1).lower()
    
    # Determine source type
    if "GCE" in filename.upper():
        metadata["source"] = "gce_official"
    elif "Sec4" in filename or "SA2" in filename:
        metadata["source"] = "school_paper"
    else:
        metadata["source"] = "unknown"
    
    return metadata


def should_skip_file(filename: str) -> bool:
    """Check if a file should be skipped during sync.
    
    Skips answer sheets and files without clear Paper1/Paper2 designation.
    """
    filename_lower = filename.lower()
    
    # Skip answer sheets
    if "_ans" in filename_lower or "_answer" in filename_lower:
        return True
    
    # Skip if no Paper designation (could be combined/ambiguous)
    if "paper" not in filename_lower:
        return True
    
    return False


def detect_section(text: str) -> Optional[str]:
    """Detect which section a chunk of text belongs to.
    
    Returns section_a, section_b, section_c, or None if unclear.
    """
    text_lower = text.lower()
    
    # Look for section markers
    if re.search(r"section\s*a\s*[\[\(]", text_lower):
        return "section_a"
    if re.search(r"section\s*b\s*[\[\(]", text_lower):
        return "section_b"
    if re.search(r"section\s*c\s*[\[\(]", text_lower):
        return "section_c"
    
    # Contextual detection based on content
    if "editing" in text_lower and "grammatical" in text_lower:
        return "section_a"
    if "situational writing" in text_lower or "write an email" in text_lower:
        return "section_b"
    if "continuous writing" in text_lower or "write a composition" in text_lower:
        return "section_c"
    if "visual text" in text_lower or "advertisement" in text_lower:
        return "section_a"  # Paper 2 Section A
    if "summary" in text_lower and "words" in text_lower:
        return "section_c"  # Paper 2 Section C
    
    return None


def chunk_text(
    text: str,
    *,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """Split text into overlapping chunks.
    
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    chunk_size = chunk_size or settings.embedding_chunk_size
    chunk_overlap = chunk_overlap or settings.embedding_chunk_overlap
    
    if not text.strip():
        return []
    
    # Split by paragraphs first, then by sentences if needed
    paragraphs = re.split(r"\n\s*\n", text)
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_length = len(para)
        
        # If single paragraph exceeds chunk size, split by sentences
        if para_length > chunk_size:
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if current_length + len(sentence) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_chunk)[-chunk_overlap:]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        else:
            # Add whole paragraph
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)[-chunk_overlap:]
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(para)
            current_length += para_length + 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Filter out very short chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
    
    return chunks


def create_paper_chunks(
    text: str,
    source_file: str,
    metadata: Dict[str, Any],
) -> List[PaperChunk]:
    """Create PaperChunk objects from text with metadata.
    
    Attempts to detect sections and assign appropriate metadata.
    """
    raw_chunks = chunk_text(text)
    
    paper_type = metadata.get("paper_type", "unknown")
    year = metadata.get("year")
    
    paper_chunks: List[PaperChunk] = []
    
    for idx, chunk_content in enumerate(raw_chunks):
        # Try to detect section for this chunk
        section = detect_section(chunk_content)
        
        paper_chunks.append(PaperChunk(
            content=chunk_content,
            paper_type=paper_type,
            section=section,
            year=year,
            source_file=source_file,
            chunk_index=idx,
            metadata={
                **metadata,
                "detected_section": section,
            },
        ))
    
    return paper_chunks


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI.
    
    Uses text-embedding-3-small model.
    """
    if not texts:
        return []
    
    client = get_openai_client()
    
    # OpenAI allows up to 2048 texts per request
    batch_size = 100
    all_embeddings: List[List[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Generated embeddings for batch {i // batch_size + 1}")
            
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
    
    return all_embeddings


def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    embeddings = generate_embeddings([query])
    if not embeddings:
        raise EmbeddingError("Failed to generate query embedding")
    return embeddings[0]

