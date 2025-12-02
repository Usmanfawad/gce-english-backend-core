"""Database module for Supabase REST API operations."""

from app.db.supabase import (
    get_supabase_client,
    init_pgvector_extension,
    create_embeddings_table,
    store_embeddings,
    search_similar_chunks,
    get_embedding_stats,
    clear_embeddings,
    PaperChunk,
    EmbeddingRecord,
    SupabaseError,
)

__all__ = [
    "get_supabase_client",
    "init_pgvector_extension",
    "create_embeddings_table",
    "store_embeddings",
    "search_similar_chunks",
    "get_embedding_stats",
    "clear_embeddings",
    "PaperChunk",
    "EmbeddingRecord",
    "SupabaseError",
]

