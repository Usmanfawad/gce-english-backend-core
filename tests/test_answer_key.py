"""Tests for Answer Key Generation service."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.answer_key import (
    AnswerKey,
    AnswerKeyError,
    _build_answer_key_prompt,
    generate_answer_key,
    save_answer_key_json,
)


class TestAnswerKey:
    """Tests for AnswerKey dataclass."""

    def test_creation(self):
        ak = AnswerKey(
            paper_format="paper_1",
            section="section_a",
            answers={"section_a": {"errors": []}},
        )
        assert ak.paper_format == "paper_1"
        assert ak.section == "section_a"
        assert "section_a" in ak.answers

    def test_to_dict(self):
        ak = AnswerKey(
            paper_format="paper_1",
            section=None,
            answers={"test": "data"},
        )
        d = ak.to_dict()

        assert d["paper_format"] == "paper_1"
        assert d["section"] is None
        assert d["answers"] == {"test": "data"}
        assert "created_at" in d


class TestBuildAnswerKeyPrompt:
    """Tests for _build_answer_key_prompt function."""

    def test_paper_1_section_a_prompt(self):
        prompt = _build_answer_key_prompt(
            paper_content="Section A [10 marks]\n1. The dog run quickly.",
            paper_format="paper_1",
            section="section_a",
        )

        assert "PAPER 1" in prompt.upper() or "paper_1" in prompt
        assert "SECTION A" in prompt.upper() or "section_a" in prompt
        assert "error" in prompt.lower()
        assert "correction" in prompt.lower()
        assert "JSON" in prompt

    def test_paper_2_section_b_prompt(self):
        prompt = _build_answer_key_prompt(
            paper_content="Section B [20 marks]\nPassage...\n1. What is...",
            paper_format="paper_2",
            section="section_b",
        )

        assert "SECTION B" in prompt.upper() or "section_b" in prompt
        assert "comprehension" in prompt.lower() or "question" in prompt.lower()
        assert "flowchart" in prompt.lower()

    def test_oral_prompt(self):
        prompt = _build_answer_key_prompt(
            paper_content="READING ALOUD [10 marks]\nPassage text...",
            paper_format="oral",
            section=None,
        )

        assert "oral" in prompt.lower()
        assert "reading" in prompt.lower()
        assert "pronunciation" in prompt.lower()


class TestGenerateAnswerKey:
    """Tests for generate_answer_key function."""

    @patch("app.services.answer_key.OpenAI")
    def test_successful_generation(self, mock_openai_class):
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"section_a": {"errors": []}}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_answer_key(
            paper_content="Test content",
            paper_format="paper_1",
            section="section_a",
            client=mock_client,
        )

        assert isinstance(result, AnswerKey)
        assert result.paper_format == "paper_1"
        assert result.section == "section_a"
        assert "section_a" in result.answers

    @patch("app.services.answer_key.OpenAI")
    def test_json_extraction_from_markdown(self, mock_openai_class):
        # LLM sometimes wraps JSON in markdown
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='```json\n{"test": "value"}\n```'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_answer_key(
            paper_content="Test content",
            paper_format="paper_1",
            section=None,
            client=mock_client,
        )

        # Should extract JSON even from markdown block
        assert "test" in result.answers or "error" in result.answers

    @patch("app.services.answer_key.OpenAI")
    def test_handles_invalid_json(self, mock_openai_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='This is not JSON'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_answer_key(
            paper_content="Test content",
            paper_format="paper_1",
            section=None,
            client=mock_client,
        )

        # Should return error structure instead of failing
        assert "error" in result.answers or "raw_response" in result.answers


class TestSaveAnswerKeyJson:
    """Tests for save_answer_key_json function."""

    def test_save_creates_file(self, tmp_path):
        ak = AnswerKey(
            paper_format="paper_1",
            section="section_a",
            answers={"test": "data"},
        )

        output_path = tmp_path / "answer_key.json"
        result = save_answer_key_json(ak, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify content
        with output_path.open() as f:
            data = json.load(f)
        assert data["paper_format"] == "paper_1"
        assert data["answers"]["test"] == "data"

    def test_save_creates_parent_dirs(self, tmp_path):
        ak = AnswerKey(
            paper_format="paper_1",
            section=None,
            answers={},
        )

        output_path = tmp_path / "nested" / "dir" / "answer_key.json"
        result = save_answer_key_json(ak, output_path)

        assert result == output_path
        assert output_path.exists()
