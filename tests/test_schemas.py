"""Tests for API schemas and validation."""

import pytest
from pydantic import ValidationError

from app.api.documents.schemas import (
    PaperGenerationRequest,
    PaperFormat,
    PaperSection,
    DifficultyLevel,
    AnswerKeyRequest,
    AnswerKeyResponse,
)


class TestPaperGenerationRequest:
    """Tests for PaperGenerationRequest schema."""

    def test_valid_paper_1_request(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.paper_1,
            section=PaperSection.section_a,
        )
        assert request.difficulty == DifficultyLevel.standard
        assert request.paper_format == PaperFormat.paper_1
        assert request.section == PaperSection.section_a

    def test_valid_paper_2_request(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.advanced,
            paper_format=PaperFormat.paper_2,
            section=PaperSection.section_c,
        )
        assert request.paper_format == PaperFormat.paper_2
        assert request.section == PaperSection.section_c

    def test_valid_oral_request_with_section(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.foundational,
            paper_format=PaperFormat.oral,
            section=PaperSection.reading_aloud,
        )
        assert request.paper_format == PaperFormat.oral
        assert request.section == PaperSection.reading_aloud

    def test_valid_oral_sbc_section(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.oral,
            section=PaperSection.sbc,
        )
        assert request.section == PaperSection.sbc

    def test_valid_oral_conversation_section(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.oral,
            section=PaperSection.conversation,
        )
        assert request.section == PaperSection.conversation

    def test_invalid_oral_with_paper_section(self):
        """Oral exam should not accept paper sections (A, B, C)."""
        with pytest.raises(ValidationError) as exc_info:
            PaperGenerationRequest(
                difficulty=DifficultyLevel.standard,
                paper_format=PaperFormat.oral,
                section=PaperSection.section_a,
            )
        assert "Oral exam only supports sections" in str(exc_info.value)

    def test_invalid_paper_1_with_oral_section(self):
        """Paper 1 should not accept oral sections."""
        with pytest.raises(ValidationError) as exc_info:
            PaperGenerationRequest(
                difficulty=DifficultyLevel.standard,
                paper_format=PaperFormat.paper_1,
                section=PaperSection.reading_aloud,
            )
        assert "Paper 1/2 only supports sections" in str(exc_info.value)

    def test_valid_full_paper_no_section(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.paper_1,
            section=None,
        )
        assert request.section is None

    def test_generate_answer_key_flag(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.paper_1,
            generate_answer_key=True,
        )
        assert request.generate_answer_key == True

    def test_visual_mode_options(self):
        for mode in ["embed", "text_only", "auto"]:
            request = PaperGenerationRequest(
                difficulty=DifficultyLevel.standard,
                paper_format=PaperFormat.paper_1,
                visual_mode=mode,
            )
            assert request.visual_mode == mode

    def test_topics_list(self):
        request = PaperGenerationRequest(
            difficulty=DifficultyLevel.standard,
            paper_format=PaperFormat.paper_1,
            topics=["environment", "technology", "social media"],
        )
        assert len(request.topics) == 3


class TestPaperSection:
    """Tests for PaperSection enum."""

    def test_paper_sections(self):
        assert PaperSection.section_a.value == "section_a"
        assert PaperSection.section_b.value == "section_b"
        assert PaperSection.section_c.value == "section_c"

    def test_oral_sections(self):
        assert PaperSection.reading_aloud.value == "reading_aloud"
        assert PaperSection.sbc.value == "sbc"
        assert PaperSection.conversation.value == "conversation"


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_all_levels(self):
        assert DifficultyLevel.foundational.value == "foundational"
        assert DifficultyLevel.standard.value == "standard"
        assert DifficultyLevel.advanced.value == "advanced"


class TestPaperFormat:
    """Tests for PaperFormat enum."""

    def test_all_formats(self):
        assert PaperFormat.paper_1.value == "paper_1"
        assert PaperFormat.paper_2.value == "paper_2"
        assert PaperFormat.oral.value == "oral"


class TestAnswerKeyRequest:
    """Tests for AnswerKeyRequest schema."""

    def test_valid_request(self):
        request = AnswerKeyRequest(
            paper_content="Section A content...",
            paper_format=PaperFormat.paper_1,
            section=PaperSection.section_a,
            output_format="json",
        )
        assert request.paper_format == PaperFormat.paper_1
        assert request.output_format == "json"

    def test_output_format_options(self):
        for fmt in ["json", "pdf", "both"]:
            request = AnswerKeyRequest(
                paper_content="Content",
                paper_format=PaperFormat.paper_1,
                output_format=fmt,
            )
            assert request.output_format == fmt
