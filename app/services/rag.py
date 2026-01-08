"""RAG (Retrieval-Augmented Generation) service for paper generation.

Enhanced RAG with:
- Higher similarity threshold (0.5) for better precision
- Section-specific retrieval with fallback strategy
- Relevance scoring and weighting
- Recency weighting for recent papers
- Increased context chunks (5) for better coverage
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime

from loguru import logger

from app.services.embeddings import generate_query_embedding, EmbeddingError
from app.db.supabase import search_similar_chunks, SupabaseError


# RAG Configuration
RAG_CONFIG = {
    "similarity_threshold": 0.5,  # Raised from 0.3 for better precision
    "max_context_chunks": 5,  # Increased from 3 for better coverage
    "max_chunk_chars": 1500,  # Maximum characters per chunk
    "recency_boost_years": 3,  # Boost papers from last N years
    "recency_boost_factor": 1.1,  # Multiply similarity by this for recent papers
    "section_match_boost": 1.15,  # Boost for exact section matches
}


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
    Enhanced with more specific keywords for better matching.
    """
    parts = []

    # Paper format context with enhanced keywords
    if paper_format == "paper_1":
        parts.append("GCE O-Level English Paper 1 Writing examination")
        if section == "section_a":
            parts.append("Section A Editing grammatical errors passage proofreading spelling punctuation verb tense")
        elif section == "section_b":
            parts.append("Section B Situational Writing formal email letter report speech proposal audience purpose register")
        elif section == "section_c":
            parts.append("Section C Continuous Writing composition essay narrative descriptive argumentative expository reflective")
        else:
            parts.append("Writing skills grammar situational continuous")
    elif paper_format == "paper_2":
        parts.append("GCE O-Level English Paper 2 Comprehension reading")
        if section == "section_a":
            parts.append("Section A Visual Text comprehension advertisement poster infographic inference persuasive technique")
        elif section == "section_b":
            parts.append("Section B Reading Comprehension passage questions inference vocabulary writer's craft language effect")
        elif section == "section_c":
            parts.append("Section C Summary guided comprehension paraphrasing key points own words")
        else:
            parts.append("Comprehension inference summary vocabulary analysis")
    elif paper_format == "oral":
        parts.append("GCE O-Level English Oral Communication spoken")
        if section == "reading_aloud":
            parts.append("Reading Aloud passage pronunciation fluency expression articulation")
        elif section == "sbc":
            parts.append("Stimulus-Based Conversation discussion visual prompt opinion analysis")
        elif section == "conversation":
            parts.append("General Conversation themes topics personal experience opinion")
        else:
            parts.append("Speaking oral reading conversation discussion")

    # Add topics with context
    if topics:
        topic_str = ", ".join(topics)
        parts.append(f"Topics and themes: {topic_str}")

    # Add difficulty context with descriptors
    difficulty_descriptors = {
        "foundational": "basic straightforward accessible",
        "standard": "moderate balanced typical",
        "advanced": "challenging complex sophisticated",
    }
    desc = difficulty_descriptors.get(difficulty, "")
    parts.append(f"Difficulty: {difficulty} {desc}")

    return " ".join(parts)


def _apply_relevance_scoring(
    chunks: List[Dict[str, Any]],
    target_section: Optional[str],
    target_paper_format: str,
) -> List[Dict[str, Any]]:
    """Apply relevance scoring with recency and section match boosts.

    Boosts:
    - Recent papers (within last N years): +10% similarity
    - Exact section match: +15% similarity

    Returns chunks sorted by adjusted similarity score.
    """
    current_year = datetime.now().year
    recency_cutoff = current_year - RAG_CONFIG["recency_boost_years"]

    scored_chunks = []
    for chunk in chunks:
        base_similarity = chunk.get("similarity", 0.0)
        adjusted_similarity = base_similarity

        # Recency boost
        year_str = chunk.get("year", "")
        if year_str:
            try:
                year = int(year_str)
                if year >= recency_cutoff:
                    adjusted_similarity *= RAG_CONFIG["recency_boost_factor"]
                    chunk["recency_boost"] = True
            except ValueError:
                pass

        # Section match boost
        chunk_section = chunk.get("section", "")
        if target_section and chunk_section == target_section:
            adjusted_similarity *= RAG_CONFIG["section_match_boost"]
            chunk["section_match_boost"] = True

        # Paper format exact match (slight boost)
        chunk_paper_type = chunk.get("paper_type", "")
        if chunk_paper_type == target_paper_format:
            adjusted_similarity *= 1.05
            chunk["paper_match_boost"] = True

        chunk["adjusted_similarity"] = min(adjusted_similarity, 1.0)  # Cap at 1.0
        scored_chunks.append(chunk)

    # Sort by adjusted similarity (descending)
    scored_chunks.sort(key=lambda x: x["adjusted_similarity"], reverse=True)

    return scored_chunks


def retrieve_relevant_context(
    *,
    paper_format: str,
    section: Optional[str] = None,
    topics: Optional[List[str]] = None,
    difficulty: str = "standard",
    limit: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant past paper content for RAG with enhanced scoring.

    Features:
    - Higher base threshold (0.5) for precision
    - Recency weighting for recent papers
    - Section-specific boosting
    - Graceful fallback strategy

    Args:
        paper_format: paper_1, paper_2, or oral
        section: Optional section filter
        topics: Optional list of topics to focus on
        difficulty: Difficulty level for context
        limit: Maximum chunks (default from config)
        similarity_threshold: Minimum similarity (default from config)

    Returns:
        List of relevant context chunks with metadata and scoring.
    """
    # Use config defaults
    limit = limit or RAG_CONFIG["max_context_chunks"]
    similarity_threshold = similarity_threshold or RAG_CONFIG["similarity_threshold"]

    try:
        # Build enhanced query
        query = build_rag_query(
            paper_format=paper_format,
            section=section,
            topics=topics,
            difficulty=difficulty,
        )

        logger.info(f"RAG query: {query[:200]}...")

        # Generate query embedding
        query_embedding = generate_query_embedding(query)

        # Retrieve more candidates than needed for scoring
        candidate_limit = limit * 2

        # Strategy 1: Try with exact section filter
        results = search_similar_chunks(
            query_embedding,
            paper_type=paper_format,
            section=section,
            limit=candidate_limit,
            similarity_threshold=similarity_threshold,
        )

        # Strategy 2: If few results with section, broaden search
        if len(results) < limit and section:
            logger.info(f"Only {len(results)} results with section={section}, broadening search...")
            broader_results = search_similar_chunks(
                query_embedding,
                paper_type=paper_format,
                section=None,
                limit=candidate_limit,
                similarity_threshold=similarity_threshold,
            )
            # Merge and deduplicate by source_file + content hash
            seen = {(r.source_file, hash(r.content[:100])) for r in results}
            for r in broader_results:
                key = (r.source_file, hash(r.content[:100]))
                if key not in seen:
                    results.append(r)
                    seen.add(key)

        # Strategy 3: If still few results, try lower threshold
        if len(results) < limit:
            lower_threshold = similarity_threshold * 0.7  # 30% lower
            logger.info(f"Trying lower threshold {lower_threshold:.2f}...")
            fallback_results = search_similar_chunks(
                query_embedding,
                paper_type=paper_format,
                section=None,
                limit=candidate_limit,
                similarity_threshold=lower_threshold,
            )
            seen = {(r.source_file, hash(r.content[:100])) for r in results}
            for r in fallback_results:
                key = (r.source_file, hash(r.content[:100]))
                if key not in seen:
                    results.append(r)
                    seen.add(key)

        logger.info(f"RAG search returned {len(results)} candidate results")

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

        # Apply relevance scoring
        scored_chunks = _apply_relevance_scoring(
            context_chunks,
            target_section=section,
            target_paper_format=paper_format,
        )

        # Take top N after scoring
        final_chunks = scored_chunks[:limit]

        logger.info(
            f"Retrieved {len(final_chunks)} relevant chunks for RAG "
            f"(best adjusted score: {final_chunks[0]['adjusted_similarity']:.2%})"
            if final_chunks else "No chunks retrieved"
        )

        return final_chunks

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
    tone, style, and structure guidance. Uses adjusted similarity scores.
    """
    if not chunks:
        return ""

    lines = [
        "## Reference Examples from Past Papers",
        "Use the following excerpts as reference for tone, structure, and style.",
        "Do NOT copy content directly; use them as guidance only.",
        "These are ranked by relevance to your current task.",
        ""
    ]

    max_chars = RAG_CONFIG["max_chunk_chars"]

    for i, chunk in enumerate(chunks, 1):
        year = chunk.get("year", "Unknown")
        section = chunk.get("section", "").replace("_", " ").title() if chunk.get("section") else ""
        paper_type = chunk.get("paper_type", "").replace("_", " ").title()

        # Use adjusted similarity if available, otherwise raw similarity
        similarity = chunk.get("adjusted_similarity", chunk.get("similarity", 0))

        # Build header with metadata
        header = f"### Reference {i}"
        meta_parts = []
        if year and year != "Unknown":
            meta_parts.append(year)
        if paper_type:
            meta_parts.append(paper_type)
        if section:
            meta_parts.append(section)

        if meta_parts:
            header += f" ({', '.join(meta_parts)})"

        lines.append(header)

        # Show relevance with boost indicators
        relevance_line = f"Relevance: {similarity:.0%}"
        boosts = []
        if chunk.get("recency_boost"):
            boosts.append("recent")
        if chunk.get("section_match_boost"):
            boosts.append("section match")
        if chunk.get("paper_match_boost"):
            boosts.append("paper match")
        if boosts:
            relevance_line += f" ({', '.join(boosts)})"

        lines.append(relevance_line)
        lines.append("")

        # Truncate content intelligently at sentence boundary if possible
        content = chunk["content"]
        if len(content) > max_chars:
            # Try to cut at sentence boundary
            truncated = content[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > max_chars * 0.7:  # At least 70% of content
                truncated = truncated[:last_period + 1]
            content = truncated + "..."

        lines.append(content)
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
    max_context_chunks: Optional[int] = None,
) -> str:
    """Enhance a generation prompt with RAG context.

    Retrieves relevant past paper content and appends it to the prompt
    as reference material for the LLM.

    Args:
        base_prompt: The original generation prompt
        paper_format: paper_1, paper_2, or oral
        section: Optional section filter
        topics: Optional topics to focus on
        difficulty: Difficulty level
        max_context_chunks: Maximum chunks (default from RAG_CONFIG)

    Returns:
        Enhanced prompt with RAG context appended.
    """
    # Use config default if not specified
    max_chunks = max_context_chunks or RAG_CONFIG["max_context_chunks"]

    # Retrieve relevant context with enhanced scoring
    chunks = retrieve_relevant_context(
        paper_format=paper_format,
        section=section,
        topics=topics,
        difficulty=difficulty,
        limit=max_chunks,
    )

    if not chunks:
        logger.debug("No RAG context available, using base prompt only")
        return base_prompt

    # Format context with scoring information
    context_section = format_rag_context(chunks)

    # Combine prompt with context
    enhanced_prompt = f"{base_prompt}\n\n{context_section}"

    # Log detailed info about what context was used
    logger.info(
        f"Enhanced prompt with {len(chunks)} RAG context chunks "
        f"(years: {[c.get('year', '?') for c in chunks]}, "
        f"sections: {[c.get('section', '?') for c in chunks]})"
    )

    return enhanced_prompt

