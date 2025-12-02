"""RAG (Retrieval-Augmented Generation) service for paper generation."""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from loguru import logger

from app.services.embeddings import generate_query_embedding, EmbeddingError
from app.db.supabase import search_similar_chunks, SupabaseError


class RAGError(RuntimeError):
    """Raised when RAG operations fail."""


def build_rag_query(
    *,
    paper_format: str,
    section: Optional[str],
    topics: Optional[List[str]],
    difficulty: str,
) -> str:
    """Build a query string for RAG retrieval based on generation parameters.
    
    Creates a natural language query that will match relevant past paper content.
    """
    parts = []
    
    # Paper format context
    if paper_format == "paper_1":
        parts.append("GCE O-Level English Paper 1 Writing")
        if section == "section_a":
            parts.append("Section A Editing grammatical errors passage")
        elif section == "section_b":
            parts.append("Section B Situational Writing email letter report speech")
        elif section == "section_c":
            parts.append("Section C Continuous Writing composition essay")
    elif paper_format == "paper_2":
        parts.append("GCE O-Level English Paper 2 Comprehension")
        if section == "section_a":
            parts.append("Section A Visual Text comprehension advertisement poster")
        elif section == "section_b":
            parts.append("Section B Reading Comprehension passage questions")
        elif section == "section_c":
            parts.append("Section C Summary guided comprehension")
    
    # Add topics if provided
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")
    
    # Add difficulty context
    parts.append(f"Difficulty level: {difficulty}")
    
    return " ".join(parts)


def retrieve_relevant_context(
    *,
    paper_format: str,
    section: Optional[str] = None,
    topics: Optional[List[str]] = None,
    difficulty: str = "standard",
    limit: int = 5,
    similarity_threshold: float = 0.3,  # Lowered from 0.65 for better recall
) -> List[Dict[str, Any]]:
    """Retrieve relevant past paper content for RAG.
    
    Args:
        paper_format: paper_1 or paper_2
        section: Optional section filter (section_a, section_b, section_c)
        topics: Optional list of topics to focus on
        difficulty: Difficulty level for context
        limit: Maximum number of chunks to retrieve
        similarity_threshold: Minimum similarity score
    
    Returns:
        List of relevant context chunks with metadata.
    """
    try:
        # Build query
        query = build_rag_query(
            paper_format=paper_format,
            section=section,
            topics=topics,
            difficulty=difficulty,
        )
        
        logger.info(f"RAG query: {query}")
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # First try with section filter
        results = search_similar_chunks(
            query_embedding,
            paper_type=paper_format,
            section=section,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
        
        # If no results with section filter, try without section filter
        if not results and section:
            logger.info(f"No results with section={section}, trying without section filter...")
            results = search_similar_chunks(
                query_embedding,
                paper_type=paper_format,
                section=None,  # Remove section filter
                limit=limit,
                similarity_threshold=similarity_threshold,
            )
        
        # If still no results, try without any filters
        if not results:
            logger.info("No results with paper_type filter, trying without filters...")
            results = search_similar_chunks(
                query_embedding,
                paper_type=None,
                section=None,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )
        
        logger.info(f"RAG search returned {len(results)} results")
        
        # Convert to dict format
        context_chunks = []
        for record in results:
            context_chunks.append({
                "content": record.content,
                "paper_type": record.paper_type,
                "section": record.section,
                "year": record.year,
                "source_file": record.source_file,
                "similarity": record.similarity,
            })
        
        logger.info(f"Retrieved {len(context_chunks)} relevant chunks for RAG")
        return context_chunks
    
    except EmbeddingError as exc:
        logger.warning(f"RAG embedding error (falling back to no context): {exc}")
        return []
    except SupabaseError as exc:
        logger.warning(f"RAG database error (falling back to no context): {exc}")
        return []
    except Exception as exc:
        logger.warning(f"RAG error (falling back to no context): {exc}")
        return []


def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt.
    
    Creates a structured reference section that the LLM can use for
    tone, style, and structure guidance.
    """
    if not chunks:
        return ""
    
    lines = [
        "## Reference Examples from Past Papers",
        "Use the following excerpts as reference for tone, structure, and style.",
        "Do NOT copy content directly; use them as guidance only.",
        ""
    ]
    
    for i, chunk in enumerate(chunks, 1):
        year = chunk.get("year", "Unknown")
        section = chunk.get("section", "").replace("_", " ").title() if chunk.get("section") else ""
        paper_type = chunk.get("paper_type", "").replace("_", " ").title()
        similarity = chunk.get("similarity", 0)
        
        header = f"### Reference {i}"
        if year != "Unknown":
            header += f" ({year}"
            if paper_type:
                header += f" {paper_type}"
            if section:
                header += f" {section}"
            header += ")"
        
        lines.append(header)
        lines.append(f"Relevance: {similarity:.0%}")
        lines.append("")
        lines.append(chunk["content"][:1500])  # Limit content length
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def get_rag_enhanced_prompt(
    base_prompt: str,
    *,
    paper_format: str,
    section: Optional[str] = None,
    topics: Optional[List[str]] = None,
    difficulty: str = "standard",
    max_context_chunks: int = 3,
) -> str:
    """Enhance a generation prompt with RAG context.
    
    Retrieves relevant past paper content and appends it to the prompt
    as reference material for the LLM.
    
    Args:
        base_prompt: The original generation prompt
        paper_format: paper_1 or paper_2
        section: Optional section filter
        topics: Optional topics to focus on
        difficulty: Difficulty level
        max_context_chunks: Maximum number of context chunks to include
    
    Returns:
        Enhanced prompt with RAG context appended.
    """
    # Retrieve relevant context
    chunks = retrieve_relevant_context(
        paper_format=paper_format,
        section=section,
        topics=topics,
        difficulty=difficulty,
        limit=max_context_chunks,
    )
    
    if not chunks:
        logger.debug("No RAG context available, using base prompt only")
        return base_prompt
    
    # Format context
    context_section = format_rag_context(chunks)
    
    # Combine prompt with context
    enhanced_prompt = f"{base_prompt}\n\n{context_section}"
    
    logger.info(f"Enhanced prompt with {len(chunks)} RAG context chunks")
    return enhanced_prompt

