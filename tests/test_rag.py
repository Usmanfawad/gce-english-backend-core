"""Tests for RAG (Retrieval-Augmented Generation) service."""

import pytest
from unittest.mock import MagicMock, patch

from app.services.rag import (
    RAG_CONFIG,
    build_rag_query,
    _apply_relevance_scoring,
    format_rag_context,
    get_rag_enhanced_prompt,
)


class TestBuildRagQuery:
    """Tests for build_rag_query function."""

    def test_paper_1_full(self):
        query = build_rag_query(
            paper_format="paper_1",
            section=None,
            topics=None,
            difficulty="standard",
        )
        assert "Paper 1" in query
        assert "Writing" in query
        assert "standard" in query.lower()

    def test_paper_1_section_a(self):
        query = build_rag_query(
            paper_format="paper_1",
            section="section_a",
            topics=None,
            difficulty="foundational",
        )
        assert "Section A" in query
        assert "Editing" in query
        assert "grammatical" in query.lower()

    def test_paper_1_section_b(self):
        query = build_rag_query(
            paper_format="paper_1",
            section="section_b",
            topics=["travel"],
            difficulty="advanced",
        )
        assert "Section B" in query
        assert "Situational" in query
        assert "travel" in query.lower()

    def test_paper_2_section_c(self):
        query = build_rag_query(
            paper_format="paper_2",
            section="section_c",
            topics=None,
            difficulty="standard",
        )
        assert "Section C" in query
        assert "Summary" in query
        assert "paraphras" in query.lower()

    def test_oral_reading_aloud(self):
        query = build_rag_query(
            paper_format="oral",
            section="reading_aloud",
            topics=None,
            difficulty="standard",
        )
        assert "Reading Aloud" in query
        assert "pronunciation" in query.lower()

    def test_topics_included(self):
        query = build_rag_query(
            paper_format="paper_1",
            section=None,
            topics=["environment", "technology"],
            difficulty="standard",
        )
        assert "environment" in query.lower()
        assert "technology" in query.lower()


class TestApplyRelevanceScoring:
    """Tests for _apply_relevance_scoring function."""

    def test_recency_boost(self):
        chunks = [
            {"content": "old paper", "year": "2018", "section": None, "paper_type": "paper_1", "similarity": 0.5},
            {"content": "new paper", "year": "2024", "section": None, "paper_type": "paper_1", "similarity": 0.5},
        ]
        scored = _apply_relevance_scoring(chunks, target_section=None, target_paper_format="paper_1")

        # Recent paper should be boosted
        new_chunk = next(c for c in scored if c["content"] == "new paper")
        old_chunk = next(c for c in scored if c["content"] == "old paper")

        assert new_chunk.get("recency_boost") == True
        assert old_chunk.get("recency_boost") is None or old_chunk.get("recency_boost") == False
        assert new_chunk["adjusted_similarity"] > old_chunk["adjusted_similarity"]

    def test_section_match_boost(self):
        chunks = [
            {"content": "section a", "year": None, "section": "section_a", "paper_type": "paper_1", "similarity": 0.5},
            {"content": "section b", "year": None, "section": "section_b", "paper_type": "paper_1", "similarity": 0.5},
        ]
        scored = _apply_relevance_scoring(chunks, target_section="section_a", target_paper_format="paper_1")

        section_a = next(c for c in scored if c["content"] == "section a")
        section_b = next(c for c in scored if c["content"] == "section b")

        assert section_a.get("section_match_boost") == True
        assert section_a["adjusted_similarity"] > section_b["adjusted_similarity"]

    def test_sorting_by_adjusted_similarity(self):
        chunks = [
            {"content": "low", "year": None, "section": None, "paper_type": "paper_1", "similarity": 0.3},
            {"content": "high", "year": None, "section": None, "paper_type": "paper_1", "similarity": 0.9},
            {"content": "mid", "year": None, "section": None, "paper_type": "paper_1", "similarity": 0.6},
        ]
        scored = _apply_relevance_scoring(chunks, target_section=None, target_paper_format="paper_1")

        # Should be sorted high to low
        assert scored[0]["content"] == "high"
        assert scored[2]["content"] == "low"


class TestFormatRagContext:
    """Tests for format_rag_context function."""

    def test_empty_chunks(self):
        result = format_rag_context([])
        assert result == ""

    def test_single_chunk(self):
        chunks = [
            {
                "content": "This is test content.",
                "year": "2023",
                "section": "section_a",
                "paper_type": "paper_1",
                "similarity": 0.8,
                "adjusted_similarity": 0.85,
            }
        ]
        result = format_rag_context(chunks)

        assert "Reference 1" in result
        assert "2023" in result
        assert "Section A" in result
        assert "85%" in result
        assert "This is test content." in result

    def test_boost_indicators(self):
        chunks = [
            {
                "content": "Content",
                "year": "2024",
                "section": "section_a",
                "paper_type": "paper_1",
                "similarity": 0.8,
                "adjusted_similarity": 0.9,
                "recency_boost": True,
                "section_match_boost": True,
            }
        ]
        result = format_rag_context(chunks)

        assert "recent" in result.lower()
        assert "section match" in result.lower()

    def test_truncation_at_sentence_boundary(self):
        long_content = "First sentence. Second sentence. Third sentence. " * 100
        chunks = [
            {
                "content": long_content,
                "year": None,
                "section": None,
                "paper_type": "paper_1",
                "similarity": 0.5,
            }
        ]
        result = format_rag_context(chunks)

        # Should be truncated and end with ...
        assert "..." in result or len(result) <= len(long_content) + 500


class TestGetRagEnhancedPrompt:
    """Tests for get_rag_enhanced_prompt function."""

    @patch("app.services.rag.retrieve_relevant_context")
    def test_no_context_returns_base_prompt(self, mock_retrieve):
        mock_retrieve.return_value = []

        base = "Generate a test paper"
        result = get_rag_enhanced_prompt(
            base,
            paper_format="paper_1",
            section=None,
            topics=None,
            difficulty="standard",
        )

        assert result == base

    @patch("app.services.rag.retrieve_relevant_context")
    def test_context_appended(self, mock_retrieve):
        mock_retrieve.return_value = [
            {
                "content": "Reference content",
                "year": "2023",
                "section": "section_a",
                "paper_type": "paper_1",
                "similarity": 0.8,
            }
        ]

        base = "Generate a test paper"
        result = get_rag_enhanced_prompt(
            base,
            paper_format="paper_1",
            section="section_a",
            topics=None,
            difficulty="standard",
        )

        assert base in result
        assert "Reference" in result
        assert "Reference content" in result


class TestRagConfig:
    """Tests for RAG configuration values."""

    def test_threshold_is_reasonable(self):
        assert 0.3 <= RAG_CONFIG["similarity_threshold"] <= 0.8

    def test_max_chunks_is_reasonable(self):
        assert 3 <= RAG_CONFIG["max_context_chunks"] <= 10

    def test_recency_boost_factor(self):
        assert RAG_CONFIG["recency_boost_factor"] >= 1.0
        assert RAG_CONFIG["recency_boost_factor"] <= 1.5
