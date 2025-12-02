"""Supabase database client using REST API (no direct PostgreSQL connection)."""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from loguru import logger
from supabase import create_client, Client

from app.config.settings import settings


# SQL for setting up the database (run once in Supabase SQL Editor)
SETUP_SQL = """
-- ===========================================
-- GCE English Backend - Database Setup
-- Run this ONCE in Supabase Dashboard → SQL Editor
-- ===========================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the embeddings table
CREATE TABLE IF NOT EXISTS paper_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    paper_type VARCHAR(20) NOT NULL,
    section VARCHAR(20),
    year VARCHAR(10),
    source_file VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_file, chunk_index)
);

-- 3. Create indexes for faster queries
CREATE INDEX IF NOT EXISTS paper_embeddings_paper_type_idx ON paper_embeddings (paper_type);
CREATE INDEX IF NOT EXISTS paper_embeddings_section_idx ON paper_embeddings (section);
CREATE INDEX IF NOT EXISTS paper_embeddings_source_file_idx ON paper_embeddings (source_file);

-- 4. Create similarity search function
CREATE OR REPLACE FUNCTION match_paper_embeddings(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5,
    filter_paper_type text DEFAULT NULL,
    filter_section text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    content text,
    paper_type varchar(20),
    section varchar(20),
    year varchar(10),
    source_file varchar(255),
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        pe.id,
        pe.content,
        pe.paper_type,
        pe.section,
        pe.year,
        pe.source_file,
        1 - (pe.embedding <=> query_embedding) as similarity
    FROM paper_embeddings pe
    WHERE 
        (1 - (pe.embedding <=> query_embedding)) >= match_threshold
        AND (filter_paper_type IS NULL OR pe.paper_type = filter_paper_type)
        AND (filter_section IS NULL OR pe.section = filter_section)
    ORDER BY pe.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Done! You can now use the /sync endpoint.
"""


class SupabaseError(RuntimeError):
    """Raised when Supabase operations fail."""


@dataclass
class PaperChunk:
    """A chunk of text from a past paper with metadata."""
    
    content: str
    paper_type: str  # paper_1 or paper_2
    section: Optional[str]  # section_a, section_b, section_c, or None
    year: Optional[str]
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass 
class EmbeddingRecord:
    """A stored embedding with its metadata."""
    
    id: str
    content: str
    paper_type: str
    section: Optional[str]
    year: Optional[str]
    source_file: str
    similarity: float = 0.0


# Singleton client
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get the Supabase client instance (singleton)."""
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    if not settings.supabase_url or not settings.supabase_key:
        raise SupabaseError(
            "Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_KEY."
        )
    
    _supabase_client = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase_client


def get_setup_sql() -> str:
    """Return the SQL needed to set up the database."""
    return SETUP_SQL


def _execute_sql_via_rest(sql: str) -> bool:
    """Execute raw SQL using Supabase's REST SQL endpoint.
    
    This requires a service_role key (not anon key) to work.
    Returns True if successful, False if not supported.
    """
    import httpx
    
    if not settings.supabase_url or not settings.supabase_key:
        return False
    
    # Try the Supabase SQL endpoint (available with service_role key)
    # Format: POST /rest/v1/rpc with a special function, or use the query endpoint
    
    # Method 1: Try using Supabase's built-in query execution
    # The service_role key can execute SQL via a special endpoint
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
    }
    
    # Supabase provides a /query endpoint for service role
    # This is not officially documented but works
    try:
        # Extract project ref from URL
        # https://xxxxx.supabase.co -> xxxxx
        import re
        match = re.search(r"https://([^.]+)\.supabase\.co", settings.supabase_url)
        if not match:
            return False
        
        project_ref = match.group(1)
        
        # Try the Supabase Management API SQL endpoint
        # This requires the service_role key
        sql_url = f"https://{project_ref}.supabase.co/rest/v1/rpc/exec_sql"
        
        response = httpx.post(
            sql_url,
            json={"sql_query": sql},
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            # exec_sql doesn't exist, try to create it first
            return False
        else:
            logger.debug(f"SQL execution returned {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.debug(f"SQL execution failed: {e}")
        return False


def init_pgvector_extension() -> None:
    """Verify Supabase connection works."""
    get_supabase_client()  # This will raise if credentials are invalid
    logger.info("Supabase client initialized successfully")


def create_embeddings_table() -> None:
    """Create the paper_embeddings table, or verify it exists."""
    supabase = get_supabase_client()
    
    # First, check if table already exists
    try:
        supabase.table("paper_embeddings").select("id").limit(1).execute()
        logger.info("paper_embeddings table exists and is accessible ✓")
        _verify_search_function(supabase)
        return
    except Exception as exc:
        error_msg = str(exc).lower()
        if "does not exist" not in error_msg and "pgrst205" not in error_msg:
            raise SupabaseError(f"Unexpected error checking table: {exc}") from exc
        # Table doesn't exist, try to create it
        logger.info("paper_embeddings table not found, attempting to create...")
    
    # Try to create table via REST SQL execution
    if _execute_sql_via_rest(SETUP_SQL):
        logger.info("Successfully created paper_embeddings table via REST API!")
        # Verify it worked
        try:
            supabase.table("paper_embeddings").select("id").limit(1).execute()
            logger.info("Table creation verified ✓")
            return
        except Exception:
            pass
    
    # If REST SQL didn't work, provide manual instructions
    logger.warning("Could not auto-create table. Manual setup required.")
    raise SupabaseError(
        "DATABASE_SETUP_REQUIRED\n\n"
        "Could not auto-create the table (this is normal on Supabase Free Tier).\n\n"
        "Please run the setup SQL manually in Supabase Dashboard:\n"
        "1. Go to: https://supabase.com/dashboard\n"
        "2. Select your project\n"
        "3. Go to 'SQL Editor' (left sidebar)\n"
        "4. Paste and run the SQL from /sync/setup-sql endpoint\n\n"
        "After running the SQL, call this endpoint again to verify."
    )


def _verify_search_function(supabase: Client) -> None:
    """Verify the similarity search function exists."""
    try:
        dummy_embedding = [0.0] * 1536
        supabase.rpc("match_paper_embeddings", {
            "query_embedding": dummy_embedding,
            "match_count": 1
        }).execute()
        logger.info("match_paper_embeddings function exists ✓")
    except Exception as func_exc:
        if "function" in str(func_exc).lower():
            logger.warning(
                "match_paper_embeddings function not found. "
                "Similarity search won't work until you create it. "
                "Run the full setup SQL from /sync/setup-sql"
            )


def store_embeddings(
    chunks: List[PaperChunk],
    embeddings: List[List[float]],
) -> int:
    """Store chunks with their embeddings in the database via REST API.
    
    Returns the number of records inserted/updated.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")
    
    if not chunks:
        return 0
    
    client = get_supabase_client()
    
    # Prepare records for upsert
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append({
            "content": chunk.content,
            "paper_type": chunk.paper_type,
            "section": chunk.section,
            "year": chunk.year,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata,
            "embedding": embedding,
        })
    
    try:
        # Upsert in batches (Supabase has payload limits)
        batch_size = 50
        total_stored = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            client.table("paper_embeddings").upsert(
                batch,
                on_conflict="source_file,chunk_index"
            ).execute()
            
            total_stored += len(batch)
            logger.debug(f"Stored batch {i // batch_size + 1}: {len(batch)} records")
        
        logger.info(f"Stored {total_stored} embeddings via REST API")
        return total_stored
        
    except Exception as exc:
        raise SupabaseError(f"Failed to store embeddings: {exc}") from exc


def search_similar_chunks(
    query_embedding: List[float],
    *,
    paper_type: Optional[str] = None,
    section: Optional[str] = None,
    limit: int = 5,
    similarity_threshold: float = 0.7,
) -> List[EmbeddingRecord]:
    """Search for similar chunks using the match_paper_embeddings RPC function."""
    client = get_supabase_client()
    
    logger.debug(
        f"Searching embeddings: paper_type={paper_type}, section={section}, "
        f"limit={limit}, threshold={similarity_threshold}"
    )
    
    try:
        result = client.rpc(
            "match_paper_embeddings",
            {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": limit,
                "filter_paper_type": paper_type,
                "filter_section": section,
            }
        ).execute()
        
        logger.debug(f"RPC returned {len(result.data or [])} results")
        
        records = []
        for row in result.data or []:
            records.append(EmbeddingRecord(
                id=row["id"],
                content=row["content"],
                paper_type=row["paper_type"],
                section=row.get("section"),
                year=row.get("year"),
                source_file=row["source_file"],
                similarity=row.get("similarity", 0.0),
            ))
        
        if records:
            logger.info(f"Found {len(records)} similar chunks (best similarity: {records[0].similarity:.2f})")
        
        return records
        
    except Exception as exc:
        error_msg = str(exc).lower()
        if "function" in error_msg and "does not exist" in error_msg:
            logger.warning("match_paper_embeddings function not found, returning empty results")
            return []
        logger.error(f"Search failed: {exc}")
        raise SupabaseError(f"Failed to search embeddings: {exc}") from exc


def get_embedding_stats() -> Dict[str, Any]:
    """Get statistics about stored embeddings via REST API."""
    client = get_supabase_client()
    
    try:
        # Get total count
        count_result = client.table("paper_embeddings").select("id", count="exact").execute()
        total_chunks = count_result.count or 0
        
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "breakdown": [],
            }
        
        # Get distinct source files
        files_result = client.table("paper_embeddings").select("source_file").execute()
        unique_files = set(r["source_file"] for r in (files_result.data or []))
        total_files = len(unique_files)
        
        # Get breakdown by paper_type and section
        all_records = client.table("paper_embeddings").select("paper_type,section").execute()
        
        breakdown_counts: Dict[tuple, int] = {}
        for record in (all_records.data or []):
            key = (record.get("paper_type"), record.get("section"))
            breakdown_counts[key] = breakdown_counts.get(key, 0) + 1
        
        # Sort with None-safe key (replace None with empty string for sorting)
        sorted_items = sorted(
            breakdown_counts.items(),
            key=lambda x: (x[0][0] or "", x[0][1] or "")
        )
        breakdown = [
            {"paper_type": k[0], "section": k[1], "count": v}
            for k, v in sorted_items
        ]
        
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "breakdown": breakdown,
        }
        
    except Exception as exc:
        error_msg = str(exc).lower()
        if "does not exist" in error_msg:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "breakdown": [],
                "note": "Table not created yet - run /sync/init-db first",
            }
        raise SupabaseError(f"Failed to get stats: {exc}") from exc


def clear_embeddings(source_file: Optional[str] = None) -> int:
    """Clear embeddings via REST API, optionally filtered by source file."""
    client = get_supabase_client()
    
    try:
        if source_file:
            delete_result = client.table("paper_embeddings").delete().eq(
                "source_file", source_file
            ).execute()
        else:
            # Delete all
            delete_result = client.table("paper_embeddings").delete().neq(
                "id", "00000000-0000-0000-0000-000000000000"
            ).execute()
        
        deleted = len(delete_result.data) if delete_result.data else 0
        logger.info(f"Deleted {deleted} embeddings")
        return deleted
        
    except Exception as exc:
        raise SupabaseError(f"Failed to clear embeddings: {exc}") from exc
