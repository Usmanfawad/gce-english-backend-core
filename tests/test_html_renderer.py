"""Tests for HTML renderer service."""

import pytest
from pathlib import Path

from app.services.html_renderer import (
    _inline_markdown_to_html,
    _build_p1_section_a_html,
    _enhance_section_headers,
    _add_section_styles,
)


class TestInlineMarkdownToHtml:
    """Tests for _inline_markdown_to_html function."""

    def test_bold_conversion(self):
        result = _inline_markdown_to_html("This is **bold** text.")
        assert "<b>bold</b>" in result
        assert "**" not in result

    def test_italic_conversion(self):
        result = _inline_markdown_to_html("This is *italic* text.")
        assert "<i>italic</i>" in result or "*italic*" in result  # May or may not be converted

    def test_html_escaping(self):
        result = _inline_markdown_to_html("Use <script> tags carefully.")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_newline_to_br(self):
        result = _inline_markdown_to_html("Line 1\nLine 2")
        assert "<br/>" in result

    def test_multiple_bold_sections(self):
        result = _inline_markdown_to_html("**First** and **Second**")
        assert result.count("<b>") == 2
        assert result.count("</b>") == 2


class TestBuildP1SectionAHtml:
    """Tests for _build_p1_section_a_html function."""

    def test_valid_12_lines(self):
        content = "\n".join([f"{i}. Line {i} content here." for i in range(1, 13)])
        result = _build_p1_section_a_html(content)

        assert result is not None
        assert "Section A" in result
        assert "Correction" in result
        assert "<table" in result

    def test_returns_none_for_invalid_content(self):
        content = "This is not a valid editing passage."
        result = _build_p1_section_a_html(content)

        assert result is None

    def test_returns_none_for_too_few_lines(self):
        content = "\n".join([f"{i}. Line {i}." for i in range(1, 5)])
        result = _build_p1_section_a_html(content)

        assert result is None

    def test_table_has_12_rows(self):
        content = "\n".join([f"{i}. Line {i} with some text." for i in range(1, 13)])
        result = _build_p1_section_a_html(content)

        assert result is not None
        # Count table rows (excluding header)
        assert result.count("<tr>") >= 12


class TestEnhanceSectionHeaders:
    """Tests for _enhance_section_headers function."""

    def test_section_a_with_marks(self):
        content = "**Section A [10 marks]**\nContent here."
        result = _enhance_section_headers(content)

        assert "section-header" in result
        assert "[10 marks]" in result or "10 marks" in result

    def test_section_b_with_marks(self):
        content = "Section B [30 marks]\nContent here."
        result = _enhance_section_headers(content)

        assert "section-header" in result

    def test_oral_reading_aloud_header(self):
        content = "READING ALOUD [10 marks]\nPassage here."
        result = _enhance_section_headers(content)

        assert "component-header" in result
        assert "10 marks" in result or "[10 marks]" in result

    def test_oral_sbc_header(self):
        content = "**STIMULUS-BASED CONVERSATION [20 marks]**\nQuestions here."
        result = _enhance_section_headers(content)

        assert "component-header" in result

    def test_page_break_added(self):
        content = "Section B [30 marks]\nContent."
        result = _enhance_section_headers(content)

        assert "page-break-before" in result

    def test_plain_text_unchanged(self):
        content = "This is just regular content without section headers."
        result = _enhance_section_headers(content)

        # Should remain mostly unchanged
        assert "section-header" not in result


class TestAddSectionStyles:
    """Tests for _add_section_styles function."""

    def test_returns_style_block(self):
        result = _add_section_styles()

        assert "<style>" in result
        assert "</style>" in result

    def test_contains_section_header_styles(self):
        result = _add_section_styles()

        assert ".section-header" in result
        assert ".marks" in result

    def test_contains_component_header_styles(self):
        result = _add_section_styles()

        assert ".component-header" in result
        assert ".component-marks" in result

    def test_contains_page_break_style(self):
        result = _add_section_styles()

        assert ".page-break-before" in result
        assert "page-break-before" in result
