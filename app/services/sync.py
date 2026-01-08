"""Sync service for processing original papers into embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from app.config.settings import settings
from app.services.ocr import extract_text_from_pdf, OCRExtractionError
from app.services.embeddings import (
    extract_metadata_from_filename,
    create_paper_chunks,
    generate_embeddings,
    should_skip_file,
)
from app.db.supabase import (
    init_pgvector_extension,
    create_embeddings_table,
    store_embeddings,
    get_embedding_stats,
)


class SyncError(RuntimeError):
    """Raised when sync operations fail."""


@dataclass
class SyncFileResult:
    """Result of processing a single file."""
    
    filename: str
    status: str  # "success", "skipped", "error"
    paper_type: Optional[str] = None
    year: Optional[str] = None
    chunks_created: int = 0
    error_message: Optional[str] = None


@dataclass
class SyncResult:
    """Result of a full sync operation."""
    
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    file_results: List[SyncFileResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "file_results": [
                {
                    "filename": fr.filename,
                    "status": fr.status,
                    "paper_type": fr.paper_type,
                    "year": fr.year,
                    "chunks_created": fr.chunks_created,
                    "error_message": fr.error_message,
                }
                for fr in self.file_results
            ],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
        }


def _get_text_output_path(pdf_path: Path) -> Path:
    """Generate the output path for OCR text from a PDF."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    stem = pdf_path.stem
    return settings.ocr_output_dir / f"{stem}-{timestamp}.txt"


def _find_existing_text(pdf_stem: str) -> Optional[Path]:
    """Find existing OCR text file for a PDF."""
    for txt_path in settings.ocr_output_dir.glob(f"{pdf_stem}*.txt"):
        return txt_path
    return None


def ensure_database_setup() -> None:
    """Ensure the database is set up with pgvector and required tables."""
    logger.info("Setting up database...")
    init_pgvector_extension()
    create_embeddings_table()
    logger.info("Database setup complete")


def process_single_file(
    file_path: Path,
    *,
    force_reprocess: bool = False,
) -> SyncFileResult:
    """Process a single PDF or text file into embeddings.
    
    Args:
        file_path: Path to the PDF or text file.
        force_reprocess: If True, reprocess even if text already exists.
    
    Returns:
        SyncFileResult with processing details.
    """
    filename = file_path.name
    result = SyncFileResult(filename=filename, status="pending")
    
    # Check if file should be skipped (answer sheets, ambiguous files)
    if should_skip_file(filename):
        result.status = "skipped"
        result.error_message = "Skipped: Answer sheet or no clear Paper1/Paper2 designation"
        logger.info(f"Skipping {filename}: answer sheet or ambiguous")
        return result
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(filename)
    result.paper_type = metadata.get("paper_type")
    result.year = metadata.get("year")
    
    # Skip if no paper type detected
    if not result.paper_type:
        result.status = "skipped"
        result.error_message = "Skipped: Could not determine Paper 1 or Paper 2"
        logger.info(f"Skipping {filename}: no paper type detected")
        return result
    
    try:
        # Determine if we need OCR or already have text
        if file_path.suffix.lower() == ".pdf":
            # Check for existing text
            existing_text = _find_existing_text(file_path.stem)
            
            if existing_text and not force_reprocess:
                logger.info(f"Using existing text for {filename}: {existing_text}")
                text = existing_text.read_text(encoding="utf-8")
                source_file = existing_text.name
            else:
                # Run OCR
                logger.info(f"Running OCR on {filename}")
                pdf_bytes = file_path.read_bytes()
                output_path = _get_text_output_path(file_path)
                ocr_result = extract_text_from_pdf(pdf_bytes, output_path)
                text = ocr_result.text
                source_file = output_path.name
                logger.info(f"OCR complete: {ocr_result.page_count} pages, {len(text)} chars")
        
        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
            source_file = filename
        
        else:
            result.status = "skipped"
            result.error_message = f"Unsupported file type: {file_path.suffix}"
            return result
        
        if not text.strip():
            result.status = "skipped"
            result.error_message = "Empty text content"
            return result
        
        # Create chunks
        chunks = create_paper_chunks(text, source_file, metadata)
        result.chunks_created = len(chunks)
        
        if not chunks:
            result.status = "skipped"
            result.error_message = "No chunks created from text"
            return result
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks from {filename}")
        chunk_texts = [c.content for c in chunks]
        embeddings = generate_embeddings(chunk_texts)
        
        # Store in database
        stored = store_embeddings(chunks, embeddings)
        
        result.status = "success"
        logger.info(f"Successfully processed {filename}: {stored} embeddings stored")
        
    except OCRExtractionError as exc:
        result.status = "error"
        result.error_message = f"OCR failed: {exc}"
        logger.error(f"OCR error for {filename}: {exc}")
    
    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.error(f"Error processing {filename}: {exc}")
    
    return result


def sync_original_papers(
    *,
    force_reprocess: bool = False,
    file_filter: Optional[str] = None,
) -> SyncResult:
    """Sync all papers from the original_papers directory.
    
    Args:
        force_reprocess: If True, reprocess PDFs even if text exists.
        file_filter: Optional glob pattern to filter files (e.g., "*Paper-1*").
    
    Returns:
        SyncResult with full processing details.
    """
    result = SyncResult()
    
    # Ensure database is set up
    ensure_database_setup()
    
    # Find all PDF files in original_papers directory
    papers_dir = settings.original_papers_dir
    if not papers_dir.exists():
        raise SyncError(f"Original papers directory not found: {papers_dir}")
    
    pattern = file_filter or "*.pdf"
    pdf_files = list(papers_dir.glob(pattern))
    
    # Also check for already-extracted text files
    if not file_filter or ".txt" in file_filter:
        txt_pattern = file_filter.replace(".pdf", ".txt") if file_filter else "*.txt"
        txt_files = list(settings.ocr_output_dir.glob(txt_pattern))
    else:
        txt_files = []
    
    all_files = pdf_files + txt_files
    result.total_files = len(all_files)
    
    if not all_files:
        logger.warning(f"No files found in {papers_dir} matching pattern '{pattern}'")
        result.completed_at = datetime.utcnow()
        return result
    
    logger.info(f"Found {len(all_files)} files to process")
    
    for file_path in all_files:
        file_result = process_single_file(file_path, force_reprocess=force_reprocess)
        result.file_results.append(file_result)
        
        if file_result.status == "success":
            result.processed_files += 1
            result.total_chunks += file_result.chunks_created
            result.total_embeddings += file_result.chunks_created
        elif file_result.status == "skipped":
            result.skipped_files += 1
        else:
            result.failed_files += 1
    
    result.completed_at = datetime.utcnow()
    
    # Log summary
    logger.info(
        f"Sync complete: {result.processed_files} processed, "
        f"{result.skipped_files} skipped, {result.failed_files} failed, "
        f"{result.total_embeddings} total embeddings"
    )
    
    # Cleanup temporary directories after processing
    _cleanup_temp_directories()
    
    return result


def _cleanup_temp_directories() -> None:
    """Remove all files from tmp/, texts/, and visuals/ directories after sync."""
    import shutil
    
    dirs_to_clean = [
        settings.temp_dir,           # storage/tmp/
        settings.ocr_output_dir,     # storage/texts/
        settings.visual_output_dir,  # storage/visuals/
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                # Remove all contents but keep the directory
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"Cleaned up directory: {dir_path}")
            except Exception as exc:
                logger.warning(f"Failed to clean {dir_path}: {exc}")


def get_sync_status() -> Dict[str, Any]:
    """Get current status of synced embeddings."""
    stats = get_embedding_stats()
    
    # Count files in original_papers
    papers_dir = settings.original_papers_dir
    pdf_count = len(list(papers_dir.glob("*.pdf"))) if papers_dir.exists() else 0
    
    # Count text files
    txt_count = len(list(settings.ocr_output_dir.glob("*.txt")))
    
    return {
        "original_papers_count": pdf_count,
        "extracted_texts_count": txt_count,
        "embedding_stats": stats,
    }

