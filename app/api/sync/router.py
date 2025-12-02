"""Sync API endpoints for processing papers into embeddings."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.sync import (
    sync_original_papers,
    get_sync_status,
    ensure_database_setup,
    SyncError,
)
from app.db.supabase import clear_embeddings, get_setup_sql, SupabaseError


router = APIRouter(prefix="/sync", tags=["sync"])


class SyncRequest(BaseModel):
    """Request body for sync operation."""
    
    force_reprocess: bool = Field(
        default=False,
        description="If true, reprocess PDFs even if text already exists",
    )
    file_filter: Optional[str] = Field(
        default=None,
        description="Optional glob pattern to filter files (e.g., '*Paper-1*')",
    )


class SyncResponse(BaseModel):
    """Response from sync operation."""
    
    total_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    total_chunks: int
    total_embeddings: int
    duration_seconds: Optional[float]
    file_results: list


class SyncStatusResponse(BaseModel):
    """Response from sync status endpoint."""
    
    original_papers_count: int
    extracted_texts_count: int
    embedding_stats: dict


@router.post("", response_model=SyncResponse)
async def sync_papers(request: SyncRequest = SyncRequest()) -> SyncResponse:
    """Sync all papers from original_papers directory.
    
    This endpoint:
    1. Scans the `storage/original_papers/` directory for PDFs
    2. Runs OCR on each PDF to extract text (or uses existing text if available)
    3. Extracts metadata (year, paper type) from filenames
    4. Chunks the text into smaller segments
    5. Generates embeddings using OpenAI text-embedding-3-small
    6. Stores everything in Supabase with pgvector
    
    The operation can take several minutes for many files.
    """
    try:
        result = sync_original_papers(
            force_reprocess=request.force_reprocess,
            file_filter=request.file_filter,
        )
        
        return SyncResponse(
            total_files=result.total_files,
            processed_files=result.processed_files,
            skipped_files=result.skipped_files,
            failed_files=result.failed_files,
            total_chunks=result.total_chunks,
            total_embeddings=result.total_embeddings,
            duration_seconds=(
                (result.completed_at - result.started_at).total_seconds()
                if result.completed_at else None
            ),
            file_results=[
                {
                    "filename": fr.filename,
                    "status": fr.status,
                    "paper_type": fr.paper_type,
                    "year": fr.year,
                    "chunks_created": fr.chunks_created,
                    "error_message": fr.error_message,
                }
                for fr in result.file_results
            ],
        )
    
    except SyncError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except SupabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database error: {exc}. Check Supabase credentials.",
        ) from exc


@router.get("/status", response_model=SyncStatusResponse)
async def get_status() -> SyncStatusResponse:
    """Get current sync status and embedding statistics."""
    try:
        status = get_sync_status()
        return SyncStatusResponse(
            original_papers_count=status["original_papers_count"],
            extracted_texts_count=status["extracted_texts_count"],
            embedding_stats=status["embedding_stats"],
        )
    except SupabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database error: {exc}. Check Supabase credentials.",
        ) from exc


@router.post("/init-db")
async def initialize_database():
    """Initialize/verify the database setup.
    
    If the table doesn't exist, returns the SQL you need to run in Supabase Dashboard.
    """
    try:
        ensure_database_setup()
        return {"status": "success", "message": "Database is ready! You can now use /sync"}
    except SupabaseError as exc:
        error_msg = str(exc)
        if "DATABASE_SETUP_REQUIRED" in error_msg:
            # Return the setup SQL for the user
            return {
                "status": "setup_required",
                "message": "Please run the setup SQL in Supabase Dashboard → SQL Editor",
                "instructions": [
                    "1. Go to https://supabase.com/dashboard",
                    "2. Select your project",
                    "3. Click 'SQL Editor' in the left sidebar",
                    "4. Paste the SQL below and click 'Run'",
                    "5. Call this endpoint again to verify"
                ],
                "setup_sql": get_setup_sql(),
            }
        raise HTTPException(
            status_code=503,
            detail=f"Database error: {exc}",
        ) from exc


@router.get("/setup-sql")
async def get_database_setup_sql():
    """Get the SQL needed to set up the database.
    
    Copy this SQL and run it in Supabase Dashboard → SQL Editor.
    """
    return {
        "instructions": "Run this SQL in Supabase Dashboard → SQL Editor",
        "sql": get_setup_sql(),
    }


@router.delete("/embeddings")
async def clear_all_embeddings(
    source_file: Optional[str] = Query(
        default=None,
        description="Optional: clear only embeddings from this source file",
    ),
):
    """Clear embeddings from the database.
    
    Use with caution - this deletes stored embeddings.
    """
    try:
        deleted = clear_embeddings(source_file)
        return {
            "status": "success",
            "deleted_count": deleted,
            "source_file": source_file,
        }
    except SupabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database error: {exc}. Check Supabase credentials.",
        ) from exc

