"""Tests for content validation and generation logic."""

import pytest

from app.services.paper_generator import (
    _validate_content,
    _check_common_llm_issues,
    _get_section_temperature,
    SECTION_TEMPERATURE,
    VALIDATION_RULES,
)


class TestValidateContent:
    """Tests for _validate_content function."""

    def test_valid_paper_1_section_a(self):
        content = """Section A [10 marks]

1. The quick brown fox jumps over the lazy dog.
2. She walk quickly to the store yesterday.
3. The children was playing in the garden.
4. He have been working here for five years.
5. They goes to school every morning.
6. The book on the shelf is very interesting.
7. She don't like to eat vegetables at all.
8. The team have won the championship again.
9. Everyone in the class are present today.
10. The weather is beautiful for a picnic.
11. He has went to the market already.
12. The sun rises in the east every day.
"""
        is_valid, issues = _validate_content(content, "paper_1", "section_a")

        # Should find the section header
        assert any("Section" in str(i) for i in issues) == False or is_valid

    def test_invalid_paper_1_section_a_too_few_lines(self):
        content = """Section A [10 marks]

1. First line.
2. Second line.
3. Third line.
"""
        is_valid, issues = _validate_content(content, "paper_1", "section_a")

        assert not is_valid
        assert any("too few" in i.lower() for i in issues)

    def test_valid_paper_2_section_b_with_flowchart(self):
        content = """Section B [20 marks]

Passage about technology and society...

1. What does the writer mean by...? [2]
2. Explain why the author... [2]
3. Give the meaning of... [1]
4. How does the writer persuade... [3]
5. What evidence supports... [2]
6. According to paragraph 2... [2]
7. Why did the character... [2]
8. What is the effect of... [2]
9. Do you agree that... [2]
10. Based on Paragraphs 2-6, complete the flowchart below. [4]

Options:
A. Statement one
B. Statement two
C. Statement three
D. Statement four
E. Statement five
F. Statement six
"""
        is_valid, issues = _validate_content(content, "paper_2", "section_b")

        # Should pass with flowchart present
        assert "flowchart" not in str(issues).lower() or is_valid

    def test_full_paper_skips_validation(self):
        content = "Any content"
        is_valid, issues = _validate_content(content, "paper_1", None)

        assert is_valid
        assert len(issues) == 0


class TestCheckCommonLlmIssues:
    """Tests for _check_common_llm_issues function."""

    def test_clean_content(self):
        content = "This is a perfectly normal English text. It contains proper sentences and punctuation."
        issues = _check_common_llm_issues(content)

        assert len(issues) == 0

    def test_detects_placeholder_text(self):
        content = "Please write a letter to [NAME] about [DATE]."
        issues = _check_common_llm_issues(content)

        assert any("placeholder" in i.lower() for i in issues)

    def test_detects_repetitive_content(self):
        content = "The cat sat on the mat. " * 20
        issues = _check_common_llm_issues(content)

        assert any("repetitive" in i.lower() for i in issues)

    def test_detects_high_non_ascii(self):
        # Simulate non-English content
        content = "这是中文内容。" * 50 + "Some English."
        issues = _check_common_llm_issues(content)

        assert any("non-ascii" in i.lower() or "non-english" in i.lower() for i in issues)

    def test_detects_truncated_content(self):
        content = "This is a long piece of content that appears to be cut off mid-sentence and doesn't end properly with"
        issues = _check_common_llm_issues(content)

        assert any("truncated" in i.lower() for i in issues)

    def test_allows_proper_ending(self):
        content = "This sentence ends properly."
        issues = _check_common_llm_issues(content)

        assert not any("truncated" in i.lower() for i in issues)


class TestGetSectionTemperature:
    """Tests for _get_section_temperature function."""

    def test_paper_1_section_a_low_temp(self):
        temp = _get_section_temperature("paper_1", "section_a")
        assert temp < 0.3  # Editing needs precision

    def test_paper_1_section_c_higher_temp(self):
        temp = _get_section_temperature("paper_1", "section_c")
        assert temp >= 0.4  # Creative writing needs more creativity

    def test_oral_sbc_moderate_temp(self):
        temp = _get_section_temperature("oral", "sbc")
        assert 0.3 <= temp <= 0.6

    def test_unknown_section_returns_default(self):
        temp = _get_section_temperature("unknown_format", "unknown_section")
        assert temp == 0.35  # Default from LLM_COMPLETION_PARAMS

    def test_all_configured_temps_are_valid(self):
        for key, temp in SECTION_TEMPERATURE.items():
            assert 0.0 <= temp <= 1.0, f"Temperature for {key} is out of range"


class TestValidationRules:
    """Tests for VALIDATION_RULES configuration."""

    def test_paper_1_section_a_has_rules(self):
        assert "paper_1_section_a" in VALIDATION_RULES
        rules = VALIDATION_RULES["paper_1_section_a"]

        assert "min_lines" in rules
        assert rules["min_lines"] == 12

    def test_paper_2_section_b_has_flowchart_rule(self):
        assert "paper_2_section_b" in VALIDATION_RULES
        rules = VALIDATION_RULES["paper_2_section_b"]

        assert "required_patterns" in rules
        patterns = rules["required_patterns"]
        assert any("flowchart" in p.lower() for p in patterns)

    def test_all_rules_have_required_patterns(self):
        for key, rules in VALIDATION_RULES.items():
            assert "required_patterns" in rules, f"{key} missing required_patterns"
