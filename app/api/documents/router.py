from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.documents.schemas import (
    DocumentIngestResponse,
    PaperGenerationRequest,
    PaperGenerationResponse,
)
from app.config.settings import settings
from app.services.ocr import OCRExtractionError, extract_text_from_pdf
from app.services.paper_generator import PaperGenerationError, generate_paper
from app.db.supabase import get_generated_paper_public_url, SupabaseError
from app.api.auth.security import require_admin_user, get_current_user
from app.services.auth import AppUser


router = APIRouter(prefix="/documents", tags=["documents"])


def _safe_stem(raw_name: str) -> str:
    stem = Path(raw_name).stem or "document"
    safe = "".join(char for char in stem if char.isalnum() or char in {"-", "_"})
    return safe or "document"


def _build_output_path(filename: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    stem = _safe_stem(filename)
    return settings.ocr_output_dir / f"{stem}-{timestamp}.txt"


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    language: str = Query("eng", description="Tesseract language code, e.g. 'eng'"),
    _current_user: AppUser = Depends(require_admin_user),
) -> DocumentIngestResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    output_path = _build_output_path(file.filename or "document.pdf")

    try:
        result = extract_text_from_pdf(pdf_bytes, output_path, language=language)
    except OCRExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        relative_output = result.output_path.relative_to(settings.storage_root)
    except ValueError:  # pragma: no cover - defensive, should not happen
        relative_output = result.output_path

    return DocumentIngestResponse(
        original_filename=file.filename or "document.pdf",
        text_path=str(relative_output),
        page_count=result.page_count,
        character_count=len(result.text),
        language=language,
    )


@router.post("/generate", response_model=PaperGenerationResponse)
async def generate_paper_endpoint(
    request: PaperGenerationRequest,
    current_user: AppUser = Depends(get_current_user),
) -> PaperGenerationResponse:
    try:
        generation_result = generate_paper(
            difficulty=request.difficulty.value,
            paper_format=request.paper_format.value,
            section=request.section.value if request.section else None,
            topics=request.topics,
            additional_instructions=request.additional_instructions,
            visual_mode=request.visual_mode or "embed",
            search_provider=request.search_provider or "openai",
            user_id=current_user.id,
            generate_answer_key_flag=request.generate_answer_key,
        )
    except PaperGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        relative_pdf = generation_result.pdf_path.relative_to(settings.storage_root)
    except ValueError:  # pragma: no cover - defensive
        relative_pdf = generation_result.pdf_path

    try:
        relative_text = generation_result.text_path.relative_to(settings.storage_root)
    except ValueError:  # pragma: no cover - defensive
        relative_text = generation_result.text_path

    preview_lines = generation_result.content.splitlines()
    preview_content = "\n".join(preview_lines[:6]).strip()

    # Handle answer key PDF path
    answer_key_pdf_str = None
    if generation_result.answer_key_pdf_path:
        try:
            answer_key_pdf_str = str(generation_result.answer_key_pdf_path.relative_to(settings.storage_root))
        except ValueError:
            answer_key_pdf_str = str(generation_result.answer_key_pdf_path)

    return PaperGenerationResponse(
        difficulty=request.difficulty,
        paper_format=request.paper_format,
        section=request.section,
        pdf_path=str(relative_pdf),
        text_path=str(relative_text),
        created_at=generation_result.created_at,
        preview=preview_content,
        visual_meta=generation_result.visual_meta,
        download_url=generation_result.download_url,
        answer_key=generation_result.answer_key,
        answer_key_pdf_path=answer_key_pdf_str,
    )


@router.get("/download-link")
async def get_generated_paper_download_link(
    file_name: str = Query(
        ...,
        description=(
            "File name of the generated PDF as stored in Supabase Storage "
            "(typically the base file name from the pdf_path, e.g. 'paper_1-section_a-standard-20251202-183000.pdf')."
        ),
    ),
) -> dict:
    """Return a public download URL for a generated paper PDF stored in Supabase Storage."""
    try:
        url = get_generated_paper_public_url(file_name)
    except SupabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {"file_name": file_name, "download_url": url}


