"""Utilities for generating synthetic exam papers using an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Tuple
import os

from loguru import logger
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.config.settings import settings
from app.db.supabase import upload_generated_paper_pdf, SupabaseError
from app.services.visuals import get_visual, VisualSnapshot
from app.services.html_renderer import html_to_pdf, render_html_template
from app.services.rag import get_rag_enhanced_prompt
from app.services.answer_key import generate_answer_key, save_answer_key_json, render_answer_key_pdf, AnswerKeyError


class PaperGenerationError(RuntimeError):
    """Raised when synthetic paper generation fails."""


@dataclass
class PaperGenerationResult:
    """Details of a generated paper."""

    content: str
    prompt: str
    pdf_path: Path
    text_path: Path
    created_at: datetime
    section: Optional[str]
    visual_meta: Optional[Dict[str, str]] = None
    download_url: Optional[str] = None
    answer_key: Optional[Dict[str, any]] = None
    answer_key_pdf_path: Optional[Path] = None
    section_a_error_key: Optional[Dict[str, any]] = None  # Pre-extracted error key for Section A


LLM_COMPLETION_PARAMS = {
    "temperature": 0.35,
    "top_p": 0.9,
    "frequency_penalty": 0.15,
    "presence_penalty": 0.0,
    "max_tokens": 6000,
}

# Section-specific temperature tuning
SECTION_TEMPERATURE = {
    # Paper 1
    "paper_1_section_a": 0.2,   # Editing - needs precision
    "paper_1_section_b": 0.4,   # Situational - moderate creativity
    "paper_1_section_c": 0.5,   # Continuous - needs prompt creativity
    # Paper 2
    "paper_2_section_a": 0.3,   # Visual text - factual
    "paper_2_section_b": 0.35,  # Comprehension - balanced
    "paper_2_section_c": 0.35,  # Summary - balanced
    # Oral
    "oral_reading_aloud": 0.4,  # Reading - moderate creativity for passage
    "oral_sbc": 0.45,           # SBC - needs engaging stimulus
    "oral_conversation": 0.4,   # Conversation - needs varied themes
}

# Content validation rules
VALIDATION_RULES = {
    "paper_1_section_a": {
        "min_lines": 12,
        "max_lines": 15,
        "required_patterns": [r"^\d+\.\s", r"Section\s*A"],  # Line numbers, section header
        "forbidden_patterns": [r"\[error\]", r"\[correction\]"],  # Don't expose errors
    },
    "paper_1_section_b": {
        "min_words": 200,
        "max_words": 500,
        "required_patterns": [r"Section\s*B", r"\d+\s*marks?"],  # Section header, marks
    },
    "paper_1_section_c": {
        "min_prompts": 4,
        "max_prompts": 4,
        "required_patterns": [r"Section\s*C", r"\d+\s*marks?"],
    },
    "paper_2_section_a": {
        "min_questions": 4,
        "required_patterns": [r"Section\s*A", r"\d+\s*marks?"],
    },
    "paper_2_section_b": {
        "min_questions": 6,
        "min_words": 500,  # Passage length (narrative)
        "required_patterns": [r"Section\s*B", r"\d+\s*marks?", r"flowchart|sequence"],
    },
    "paper_2_section_c": {
        "min_questions": 4,
        "required_patterns": [r"Section\s*C", r"summary", r"\d+\s*marks?"],
    },
}


def _get_section_temperature(paper_format: str, section: Optional[str]) -> float:
    """Get the appropriate temperature for a section."""
    if section:
        key = f"{paper_format}_{section}"
    else:
        key = paper_format
    return SECTION_TEMPERATURE.get(key, LLM_COMPLETION_PARAMS["temperature"])


def _extract_section_a_error_key(content: str) -> Tuple[str, Optional[Dict[str, any]]]:
    """
    Extract the error key from Section A content and return cleaned content + error data.

    The error key is expected in format:
    ===ERROR_KEY_START===
    Line X: "incorrect_word" should be "correct_word" (error_type)
    ...
    Correct lines: [1, 5, 12]
    ===ERROR_KEY_END===

    Returns:
        Tuple of (cleaned_content, error_key_dict or None)
    """
    import re

    error_key_pattern = r'===ERROR_KEY_START===\s*(.*?)\s*===ERROR_KEY_END==='
    match = re.search(error_key_pattern, content, re.DOTALL)

    if not match:
        logger.warning("No error key found in Section A content")
        return content, None

    error_key_text = match.group(1).strip()
    cleaned_content = re.sub(error_key_pattern, '', content, flags=re.DOTALL).strip()

    # Parse the error key
    errors = []
    correct_lines = []

    # Parse error lines: Line X: "incorrect_word" should be "correct_word" (error_type)
    error_line_pattern = r'Line\s+(\d+):\s*["\']?([^"\']+)["\']?\s+should\s+be\s+["\']?([^"\']+)["\']?\s*\(([^)]+)\)'
    for err_match in re.finditer(error_line_pattern, error_key_text, re.IGNORECASE):
        errors.append({
            "line": int(err_match.group(1)),
            "error": err_match.group(2).strip(),
            "correction": err_match.group(3).strip(),
            "error_type": err_match.group(4).strip(),
        })

    # Parse correct lines: Correct lines: [1, 5, 12] or Correct lines: 1, 5, 12
    correct_pattern = r'Correct\s+lines?:\s*\[?([^\]]+)\]?'
    correct_match = re.search(correct_pattern, error_key_text, re.IGNORECASE)
    if correct_match:
        correct_str = correct_match.group(1)
        correct_lines = [int(n.strip()) for n in re.findall(r'\d+', correct_str)]

    error_key_data = {
        "errors": errors,
        "correct_lines": correct_lines,
        "total_errors": len(errors),
    }

    logger.info(f"Extracted Section A error key: {len(errors)} errors, correct lines: {correct_lines}")

    return cleaned_content, error_key_data


def _extract_flowchart_answer_key(content: str) -> Tuple[str, Optional[Dict[str, any]]]:
    """
    Extract the flowchart answer key from Section B content and return cleaned content + answer data.

    The answer key is expected in format:
    ===FLOWCHART_ANSWER_KEY_START===
    Paragraph 2: A (reason: ...)
    ...
    Distractors: B, E
    ===FLOWCHART_ANSWER_KEY_END===

    Returns:
        Tuple of (cleaned_content, flowchart_answer_dict or None)
    """
    import re

    answer_key_pattern = r'===FLOWCHART_ANSWER_KEY_START===\s*(.*?)\s*===FLOWCHART_ANSWER_KEY_END==='
    match = re.search(answer_key_pattern, content, re.DOTALL)

    flowchart_data = None

    if match:
        answer_key_text = match.group(1).strip()
        content = re.sub(answer_key_pattern, '', content, flags=re.DOTALL).strip()

        # Parse the answer key
        answers = {}
        distractors = []

        # Parse paragraph answers: Paragraph X: Y (reason: ...)
        para_pattern = r'Paragraph\s+(\d+):\s*([A-F])\s*(?:\(reason:\s*([^)]+)\))?'
        for para_match in re.finditer(para_pattern, answer_key_text, re.IGNORECASE):
            para_num = int(para_match.group(1))
            answer = para_match.group(2).upper()
            reason = para_match.group(3).strip() if para_match.group(3) else ""
            answers[f"paragraph_{para_num}"] = {"answer": answer, "reason": reason}

        # Parse distractors: Distractors: B, E
        distractor_pattern = r'Distractors?:\s*([A-F,\s]+)'
        distractor_match = re.search(distractor_pattern, answer_key_text, re.IGNORECASE)
        if distractor_match:
            distractor_str = distractor_match.group(1)
            distractors = [d.strip().upper() for d in re.findall(r'[A-F]', distractor_str)]

        flowchart_data = {
            "answers": answers,
            "distractors": distractors,
        }

        logger.info(f"Extracted flowchart answer key: {len(answers)} answers, distractors: {distractors}")

    # Also clean up any remaining flowchart answers that might appear in student content
    # Pattern: "Paragraph X: [A-F]" without the [____] blank
    # We want to replace "Paragraph 2: A" with "Paragraph 2: [____]"
    content = _clean_flowchart_answers(content)

    return content, flowchart_data


def _clean_flowchart_answers(content: str) -> str:
    """
    Remove any flowchart answers that appear in the student content.
    Replace "Paragraph X: A" with "Paragraph X: [____]"
    """
    import re

    # Pattern to match flowchart answers like "Paragraph 2: A" or "Paragraph 3: C"
    # but NOT lines that already have blanks like "Paragraph 2: [____]"
    # Also handle variations like "│ Paragraph 2: A" in box drawings

    def replace_answer(match):
        prefix = match.group(1) if match.group(1) else ""
        para_num = match.group(2)
        return f"{prefix}Paragraph {para_num}: [____]"

    # Match "Paragraph X: [single letter A-F]" that's not followed by more text
    # This catches both plain text and box drawing formats
    pattern = r'(│\s*)?Paragraph\s+(\d+):\s*([A-F])(?:\s*│|\s*$|\s*\n)'

    def full_replace(match):
        prefix = match.group(1) if match.group(1) else ""
        para_num = match.group(2)
        suffix = "│" if match.group(0).rstrip().endswith("│") else ""
        if suffix:
            return f"{prefix}Paragraph {para_num}: [____]                     {suffix}\n"
        return f"{prefix}Paragraph {para_num}: [____]\n"

    content = re.sub(pattern, full_replace, content, flags=re.IGNORECASE)

    # Also remove any "ANSWER KEY" sections that might have slipped through without markers
    answer_key_section_pattern = r'\n*(?:ANSWER KEY|Answer Key).*?(?=\n\n[A-Z]|\n\n\*\*|\Z)'
    content = re.sub(answer_key_section_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    return content


def _validate_content(content: str, paper_format: str, section: Optional[str]) -> Tuple[bool, List[str]]:
    """Validate generated content against rules. Returns (is_valid, list_of_issues)."""
    import re

    issues = []

    if section:
        key = f"{paper_format}_{section}"
    else:
        # For full paper, validate each section
        return True, []  # Skip validation for full papers (validated per-section)

    rules = VALIDATION_RULES.get(key, {})
    if not rules:
        return True, []

    # Check word count
    word_count = len(content.split())
    if "min_words" in rules and word_count < rules["min_words"]:
        issues.append(f"Content too short: {word_count} words (min: {rules['min_words']})")
    if "max_words" in rules and word_count > rules["max_words"]:
        issues.append(f"Content too long: {word_count} words (max: {rules['max_words']})")

    # Check line count (for editing section)
    if "min_lines" in rules:
        numbered_lines = len(re.findall(r"^\d+\.\s", content, re.MULTILINE))
        if numbered_lines < rules["min_lines"]:
            issues.append(f"Too few numbered lines: {numbered_lines} (min: {rules['min_lines']})")

    # Check required patterns
    for pattern in rules.get("required_patterns", []):
        if not re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Missing required pattern: {pattern}")

    # Check forbidden patterns
    for pattern in rules.get("forbidden_patterns", []):
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Found forbidden pattern: {pattern}")

    # Check question count
    if "min_questions" in rules:
        # Count question numbers like Q1, Q2, 1., 2., (a), (b)
        question_count = len(re.findall(r"(?:^|\n)\s*(?:Q?\d+[\.\):]|\(\w\))", content))
        if question_count < rules["min_questions"]:
            issues.append(f"Too few questions: {question_count} (min: {rules['min_questions']})")

    # Check prompt count (for Section C)
    if "min_prompts" in rules:
        prompt_count = len(re.findall(r"(?:^|\n)\s*\d+\.\s+(?:Write|Describe|'|\")", content))
        if prompt_count < rules["min_prompts"]:
            issues.append(f"Too few prompts: {prompt_count} (min: {rules['min_prompts']})")

    is_valid = len(issues) == 0
    return is_valid, issues


def _check_common_llm_issues(content: str) -> List[str]:
    """Check for common LLM generation issues."""
    import re

    issues = []

    # Check for non-English content (common issue)
    non_ascii_ratio = sum(1 for c in content if ord(c) > 127) / max(len(content), 1)
    if non_ascii_ratio > 0.1:  # More than 10% non-ASCII
        issues.append("High ratio of non-ASCII characters (possible non-English content)")

    # Check for repetitive content
    words = content.lower().split()
    if len(words) > 50:
        # Check for repeated phrases (3+ word sequences)
        phrases = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        max_repeat = max(phrase_counts.values()) if phrase_counts else 0
        if max_repeat > 5:
            issues.append(f"Repetitive content detected (phrase repeated {max_repeat} times)")

    # Check for placeholder text
    placeholders = re.findall(r"\[(?:DATE|TIME|NAME|LOCATION|INSERT|TODO|TBD|XXX)\]", content, re.IGNORECASE)
    if placeholders:
        issues.append(f"Placeholder text found: {placeholders[:3]}")

    # Check for incomplete sentences at end
    content_stripped = content.strip()
    if content_stripped and content_stripped[-1] not in ".!?\"')\n":
        if len(content_stripped) > 100:  # Only flag for substantial content
            issues.append("Content may be truncated (doesn't end with proper punctuation)")

    return issues


def _build_prompt(
    *,
    difficulty: str,
    paper_format: str,
    section: Optional[str],
    topics: Optional[Iterable[str]],
    additional_instructions: Optional[str],
) -> str:
    structure_guidance = _official_structure_guidance(paper_format, section)

    friendly_format = paper_format.replace("_", " ").title()
    section_fragment = ""
    if section:
        section_fragment = f"\nTarget section: {section.replace('_', ' ').title()}."

    topic_section = ""
    if topics:
        topics_clean = ", ".join(t.strip() for t in topics if t and t.strip())
        if topics_clean:
            topic_section = f"\nFocus topics: {topics_clean}"

    instructions_section = ""
    if additional_instructions:
        instructions_section = f"\nAdditional guidance: {additional_instructions.strip()}"

    reference_excerpt = _load_reference_excerpt(paper_format)
    reference_block = ""
    if reference_excerpt:
        reference_block = (
            "\nUse the following short excerpt as a reference for tone and structure (do not copy):\n"
            f"{reference_excerpt}\n"
        )

    prompt = (
        "Generate a new GCE O-Level English examination paper.\n"
        "CRITICAL: All content must be written EXCLUSIVELY in English. Do not use any other language.\n"
        "All questions, instructions, passages, and prompts must be in English only.\n"
        f"Target difficulty: {difficulty}.\n"
        f"Paper format: {friendly_format}.{section_fragment}\n"
        f"{structure_guidance}\n"
        "Generate ONLY the requested section if a section is specified; do NOT include other sections.\n"
        "DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, INSERT, etc.). "
        "Generate ONLY the section content itself, starting with the section heading (e.g., 'Section B [30 marks]').\n"
        "Provide clearly separated sections with instructions and marking allocations "
        "when appropriate. Use numbered questions and realistic, context-rich prompts.\n"
        "Ensure the paper is coherent, internally consistent, and suitable for classroom use.\n"
        "Avoid placeholders such as [Date], [Time], [Location]. If details are unknown, omit them rather than using brackets."
        f"{topic_section}{instructions_section}\n"
        f"{reference_block}"
        "Describe any required visual stimulus in words; do not embed actual images or external links.\n"
        "Return only the paper content without extra commentary."
    )
    return prompt


def _official_structure_guidance(paper_format: str, section: Optional[str]) -> str:
    base_guidance: Dict[str, Dict[Optional[str], str]] = {
        "paper_1": {
            None: dedent(
                """\
                Follow the official Paper 1 structure and marking:
                - Section A [10 marks] (Editing): Provide a single 12-line passage. The FIRST and LAST lines are correct. Exactly EIGHT of the remaining lines (out of Lines 2–11) contain ONE grammatical error each. Exactly TWO additional lines contain NO error; place them unpredictably. Output each line prefixed by its line number (1–12). After the passage, include an “Answer Spaces” list with 12 numbered blanks for student corrections. Do not supply answers.
                - Section B [30 marks] (Situational Writing): Create ONE situational task (letter, email, report, or speech) based on a web-page style visual stimulus. Describe the visual stimulus textually (headings, callouts, short blurbs); include purpose, audience, and context. Indicate 3–5 key points the student must address. Advise 250–350 words.
                - Section C [30 marks] (Continuous Writing): Provide FOUR prompts total, covering required genres: Narrative (always), and three of {Descriptive, Expository, Argumentative, Reflective}. Instruct students to choose ONE and advise 350–500 words.
                """
            ),
            "section_a": dedent(
                """\
                Generate Paper 1 Section A [10 marks] (Editing), SINGLE SECTION ONLY.

                *** YOU MUST CREATE EXACTLY 8 GRAMMATICALLY WRONG SENTENCES ***

                FIXED ERROR PLACEMENT - FOLLOW THIS EXACTLY:
                - Line 1: CORRECT (no error)
                - Line 2: MUST HAVE ERROR
                - Line 3: MUST HAVE ERROR  
                - Line 4: CORRECT (no error)
                - Line 5: MUST HAVE ERROR
                - Line 6: MUST HAVE ERROR
                - Line 7: MUST HAVE ERROR
                - Line 8: CORRECT (no error)
                - Line 9: MUST HAVE ERROR
                - Line 10: MUST HAVE ERROR
                - Line 11: MUST HAVE ERROR
                - Line 12: CORRECT (no error)

                Total: 8 error lines (2,3,5,6,7,9,10,11) + 4 correct lines (1,4,8,12)

                ---

                OUTPUT FORMAT:

                **Section A [10 marks]**

                The following passage contains some errors. Each of the 12 lines may contain one error. If there is an error, write the correction in the space provided. If the line is correct, put a tick (✓).

                1. [CORRECT sentence - no errors]
                2. [Sentence with ONE clear grammatical error]
                3. [Sentence with ONE clear grammatical error]
                4. [CORRECT sentence - no errors]
                5. [Sentence with ONE clear grammatical error]
                6. [Sentence with ONE clear grammatical error]
                7. [Sentence with ONE clear grammatical error]
                8. [CORRECT sentence - no errors]
                9. [Sentence with ONE clear grammatical error]
                10. [Sentence with ONE clear grammatical error]
                11. [Sentence with ONE clear grammatical error]
                12. [CORRECT sentence - no errors]

                ===ERROR_KEY_START===
                Line 2: "[wrong]" should be "[correct]" (error type)
                Line 3: "[wrong]" should be "[correct]" (error type)
                Line 5: "[wrong]" should be "[correct]" (error type)
                Line 6: "[wrong]" should be "[correct]" (error type)
                Line 7: "[wrong]" should be "[correct]" (error type)
                Line 9: "[wrong]" should be "[correct]" (error type)
                Line 10: "[wrong]" should be "[correct]" (error type)
                Line 11: "[wrong]" should be "[correct]" (error type)
                Correct lines: 1, 4, 8, 12
                ===ERROR_KEY_END===

                ---

                *** USE THESE 8 ERROR TYPES - ONE PER ERROR LINE ***
                
                *** IMPORTANT: Per official syllabus, ONLY GRAMMATICAL errors are tested ***
                *** Do NOT include spelling or punctuation errors ***

                Line 2 ERROR - Subject-verb (plural subject needs plural verb):
                   WRONG: "The students was excited" or "Many people was happy"
                   CORRECT: were
                   
                Line 3 ERROR - Subject-verb (singular subject needs singular verb):
                   WRONG: "The teacher have planned" or "She have finished"
                   CORRECT: has
                   
                Line 5 ERROR - Wrong tense (past event needs past tense):
                   WRONG: "Yesterday I walk to school" or "Last week he go there"
                   CORRECT: walked, went
                   
                Line 6 ERROR - Subject-verb (third person singular needs -s):
                   WRONG: "The system allow users" or "This method provide benefits"
                   CORRECT: allows, provides
                   
                Line 7 ERROR - Wrong adverb form (adverb needed, not adjective):
                   WRONG: "She spoke very soft" or "He ran very quick"
                   CORRECT: softly, quickly
                   
                Line 9 ERROR - Wrong participle form (passive needs past participle):
                   WRONG: "should be encourage" or "must be complete"
                   CORRECT: encouraged, completed
                   
                Line 10 ERROR - Wrong word form (noun/adjective confusion):
                   WRONG: "It was a beauty day" or "The success of the plan"
                   CORRECT: beautiful, successful (when adjective needed)
                   
                Line 11 ERROR - Wrong pronoun or determiner:
                   WRONG: "Me and him went" or "Everyone brought their own"
                   CORRECT: "He and I went", "Everyone brought his or her own"

                ---

                *** COMPLETE EXAMPLE - COPY THIS STRUCTURE EXACTLY ***

                **Section A [10 marks]**

                The following passage contains some errors. Each of the 12 lines may contain one error. If there is an error, write the correction in the space provided. If the line is correct, put a tick (✓).

                1. The annual sports day at Riverside School was a memorable event for everyone.
                2. All the students was excited to participate in the various competitions.
                3. The head teacher have been planning this event since the beginning of term.
                4. Parents and teachers gathered early to find good seats near the field.
                5. The first race begin at nine o'clock sharp with the youngest students.
                6. This event bring together families from all parts of the community.
                7. The athletes ran very quick around the track in the final race.
                8. Many photographs were taken to capture the special moments of the day.
                9. The winning team was suppose to receive their medals at the ceremony.
                10. It was a beauty day and everyone enjoyed the warm sunshine.
                11. Me and my friends cheered loudly for all the participants.
                12. Students and parents left the school feeling proud and happy that evening.

                ===ERROR_KEY_START===
                Line 2: "was" should be "were" (subject-verb agreement with plural "students")
                Line 3: "have" should be "has" (subject-verb agreement with singular "teacher")
                Line 5: "begin" should be "began" (past tense required)
                Line 6: "bring" should be "brings" (third person singular needs -s)
                Line 7: "quick" should be "quickly" (adverb needed after verb)
                Line 9: "suppose" should be "supposed" (past participle needed)
                Line 10: "beauty" should be "beautiful" (adjective needed, not noun)
                Line 11: "Me" should be "My friends and I" (correct pronoun form)
                Correct lines: 1, 4, 8, 12
                ===ERROR_KEY_END===

                ---

                NOW CREATE A NEW 12-LINE PASSAGE:
                - Use a DIFFERENT topic (not sports day)
                - ERRORS MUST be in lines: 2, 3, 5, 6, 7, 9, 10, 11
                - CORRECT lines: 1, 4, 8, 12
                - Each error must be OBVIOUSLY grammatically wrong
                - Use the 8 error types listed above
                """
            ),
            "section_b": dedent(
                """\
                Generate Paper 1 Section B [30 marks] (Situational Writing), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section B content.
                - Start directly with "**Section B [30 marks]**" or "Section B [30 marks]".

                O-LEVEL SYLLABUS ALIGNMENT:
                - Tasks MUST be appropriate for 15-17 year old students taking GCE O-Level English
                - APPROPRIATE TOPICS: community events, educational programs, environmental initiatives, health campaigns, youth activities, cultural events, library/museum programs, volunteer opportunities, sports programs, school-related activities
                - AVOID TOPICS: visa/immigration, work permits, adult financial services, complex legal matters, topics beyond student experience

                VISUAL STIMULUS INTEGRATION:
                - If a visual stimulus is provided, you MUST acknowledge it in your task instructions. Begin with a statement like "You have come across a visually appealing webpage/poster/advertisement..." or "Refer to the visual stimulus provided..." or "Using the visual stimulus shown above..." to explicitly reference the image.
                - The visual should present COMPELLING, PERSUASIVE content that gives students clear reasons to choose between options or take action.
                - If the visual shows multiple options (e.g., courses, programs, destinations), ensure the task requires students to make informed comparisons.
                - If the visual contains ANY immigration, visa, or work permit related content, IGNORE those elements entirely and focus only on the community/educational aspects.

                TASK DESIGN:
                - Choose ONE appropriate task type (letter, email, report, or speech) consistent with the visual stimulus topic.
                - You MUST base the task on the provided visual description; reflect its headings/callouts/blurbs.
                - Ensure the scenario is relatable and achievable for a secondary school student.

                PAC (Purpose, Audience, Context) - IMPLICIT INTEGRATION:
                - DO NOT use explicit "Purpose:", "Audience:", "Context:" labels.
                - Instead, WEAVE the PAC elements naturally into the situational scenario. The purpose, audience, and context should be CLEAR from the narrative setup without being labeled.
                - WRONG: "Purpose: To persuade your friend. Audience: A close friend. Context: You saw an advertisement."
                - CORRECT: "Your close friend has been looking for a photography course to develop their hobby. You recently came across this advertisement and believe one of the courses would be perfect for them. Write an email to persuade them to sign up, explaining why you think it suits their interests and skill level."

                KEY POINTS & GUIDANCE:
                - List 3–5 key points students must address that directly tie to the visual description.
                - State tone/register explicitly based on the audience:
                  • FORMAL tone: For principal, teachers, official school bodies, external organisations
                  • SEMI-FORMAL tone: For school newsletter, club members, community groups
                  • INFORMAL tone: For friends, peers, family members
                - Example: "Use a formal and enthusiastic tone" (for writing to principal)
                - Advise 250–350 words.
                - Avoid placeholders like [Date]/[Time]/[Location]; if not essential, omit them.
                - End with "---" or "[End of Section B]" marker.
                """
            ),
            "section_c": dedent(
                """\
                Generate Paper 1 Section C [30 marks] (Continuous Writing), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, INSERT, etc.). Generate ONLY the Section C content.
                - Start directly with "**Section C [30 marks]**" or "Section C [30 marks]".
                - Present FOUR prompts (not five).
                
                PROMPT STYLE - NO GENRE LABELS:
                - DO NOT label prompts with genre names like "Narrative:", "Descriptive:", "Expository:", "Argumentative:", or "Reflective:".
                - The genre should be IMPLIED by the prompt's wording, not explicitly stated.
                - WRONG: "1. **Narrative**: Write a story about a time when..."
                - WRONG: "1. (Narrative) Write a story about..."
                - CORRECT: "1. Write about a time when you had to make a difficult decision. What led to this moment, and how did it change you?"
                - CORRECT: "1. 'The door creaked open slowly.' Continue this story."
                - CORRECT: "1. Describe a place that holds special meaning to you."
                - CORRECT: "1. 'Technology has made our lives easier.' Do you agree?"
                
                PROMPT CONTENT:
                - Internally ensure variety: include at least one narrative-style, one descriptive-style, one argumentative/expository-style prompt.
                - Each prompt should be authentic, concise, and may include 1–2 guiding questions or cues to help candidates develop their ideas.
                - Prompts can be questions, statements to respond to, or story starters.
                
                FORMAT:
                - Number prompts simply as "1.", "2.", "3.", "4." without genre prefixes.
                - Instruct students to choose ONE and advise 350–500 words.
                - End with "---" or "[End of Section C]" marker.
                """
            ),
        },
        "paper_2": {
            None: dedent(
                """\
                Follow the official Paper 2 structure and marking:
                - Section A [5 marks] (Visual Text Comprehension):
                  Provide ONE visual text (webpage/poster/advertisement) described fully in words (no images or links).
                  Include headings, callouts, short blurbs, and layout cues (e.g., banner, side panel).
                  Set EXACTLY 4 questions where ONE has two subparts (e.g., Q1(a), Q1(b)) so the total marks is 5.
                  Typical mix: Q1(a) Literal, Q1(b) Literal, Q2 Persuasive technique, Q3 Language effect (phrased as "How does X persuade/influence the reader..."), Q4 Inference.
                  Questions must explicitly reference elements of the described visual text.
                - Section B [20 marks] (Reading Comprehension Open-Ended - NARRATIVE):
                  *** Per official syllabus: "Text 3 which is narrative in nature" ***
                  Supply ONE NARRATIVE passage (story/recount) of about 600–650 words with characters, setting, plot, and resolution.
                  Include paragraph numbers. Set ~6 questions (Q5-Q10) covering literal retrieval, vocabulary-in-context,
                  writer's craft for narrative (tension, mood, character portrayal), and evaluation.
                  The FINAL question (Q10) MUST be a 4-mark SEQUENCE flowchart showing story events in order (not themes).
                - Section C [25 marks] (Guided Comprehension + Summary - NON-NARRATIVE):
                  *** Per official syllabus: "Text 4, which is non-narrative in nature" ***
                  Provide a NON-NARRATIVE passage (expository/argumentative/informational) around 400–550 words.
                  Before the summary, set 4–5 questions totaling 10 marks (short-answer mix relevant to the new passage).
                  Then set a 15-mark summary task covering a SUBSET of body paragraphs (e.g., Paragraphs 2-5, not the full text);
                  require continuous writing (≤80 words), using own words as far as possible (not note form).
                """
            ),
            "section_a": dedent(
                """\
                Generate Paper 2 Section A [5 marks] (Visual Text Comprehension), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section A content.
                - Start directly with "**Section A [5 marks]**" or "Section A [5 marks]".

                O-LEVEL SYLLABUS ALIGNMENT:
                - Questions MUST be appropriate for 15-17 year old students taking GCE O-Level English
                - Focus on VISUAL TEXT COMPREHENSION skills: identifying information, understanding persuasive techniques, analysing language effects
                - AVOID questions about: visa/immigration, work permits, adult financial services, complex legal matters

                VISUAL STIMULUS REQUIREMENTS:
                - Include a DETAILED DESCRIPTION of the visual stimulus with:
                  • Clear headline/title
                  • At least 3-5 specific program names, event names, or feature names (these become answers for Q1)
                  • At least 2-3 persuasive phrases or slogans (for Q2 and Q3)
                  • Statistics, dates, or factual details
                  • Call-to-action phrases
                  • Organization name and tagline
                - The visual must have ENOUGH DETAIL for all questions to be answerable

                VISUAL STIMULUS INTEGRATION:
                - Begin with: "Refer to the visual stimulus provided above and answer Questions 1-4."
                - Write questions that reference SPECIFIC elements visible in the visual.

                QUESTION REQUIREMENTS - CRITICAL:
                
                Q1(a) and Q1(b) - DIRECT RETRIEVAL [1 mark each]:
                - These MUST require students to COPY EXACT PHRASES/WORDS from the visual
                - The answer must be a direct quote, NOT a paraphrase
                - Be SPECIFIC in your question about what aspect you're asking about
                - WRONG: "What does the visual state about the nature of meetings?" (too vague)
                - CORRECT: "What does the visual state about the frequency and structure of meetings held by the network?"
                - CORRECT: "Identify the phrase that describes..." or "What is the name of..." or "State the exact phrase that..."
                - Answer must be DIRECTLY LIFTABLE from the visual text
                
                Q2 - PERSUASIVE TECHNIQUE [1 mark]:
                - Quote an EXACT phrase from the visual and ask about its persuasive effect
                - CORRECT: "Explain how the phrase '[exact quote]' serves as a persuasive technique."
                - The quoted phrase MUST appear word-for-word in the visual description
                
                Q3 - LANGUAGE EFFECT [1 mark]:
                - Quote an EXACT phrase and ask about its effect on the reader
                - CORRECT: "Explain the effect of the phrase '[exact quote]' on the reader."
                - The quoted phrase MUST appear word-for-word in the visual description
                
                Q4 - INFERENCE [1 mark]:
                - Ask what can be inferred, requiring textual evidence
                - CORRECT: "Based on the information in the visual, what can you infer about X? Provide evidence from the visual to support your answer."

                VALIDATION CHECKLIST:
                □ Visual description has at least 3 specific names/terms for Q1 answers
                □ Visual description has at least 2 quotable persuasive phrases for Q2/Q3
                □ Q1(a) answer is an EXACT phrase from the visual
                □ Q1(b) answer is an EXACT phrase from the visual
                □ Q2 quotes an EXACT phrase that EXISTS in the visual
                □ Q3 quotes an EXACT phrase that EXISTS in the visual
                □ Q4 can be answered with evidence from the visual
                """
            ),
            "section_b": dedent(
                """\
                Generate Paper 2 Section B [20 marks] (Reading Comprehension Open-Ended), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section B content.
                - Start directly with "**Section B [20 marks]**" or "Section B [20 marks]".
                
                *** OFFICIAL SYLLABUS REQUIREMENT: Text 3 MUST be NARRATIVE in nature ***
                
                PASSAGE TYPE - NARRATIVE (Story or Recount):
                - Generate a NARRATIVE passage (~600–650 words) - this is a STORY or personal RECOUNT
                - The passage should have characters, settings, events, and/or conflict
                - Examples: A personal experience, a short story, an adventure, a memoir excerpt, a biographical recount
                - NOT an expository essay, NOT an argumentative text, NOT an informational article
                - Include clear PARAGRAPH numbering (Paragraph 1, 2, 3, etc.)
                
                NARRATIVE ELEMENTS TO INCLUDE:
                - Setting: Where and when the story takes place
                - Character(s): At least one main character with some development
                - Plot: A clear sequence of events with beginning, middle, end
                - Conflict or challenge: Something the character(s) must face
                - Resolution: How things turn out
                - Descriptive language: Sensory details, imagery, figurative language
                
                QUESTION NUMBERING - CRITICAL:
                - Section A has Questions 1-4 (visual text)
                - Section B MUST start at Question 5 and continue: 5, 6, 7, 8, 9, 10
                - DO NOT restart numbering at 1

                REQUIRED QUESTION TYPES FOR NARRATIVE (Q5-Q10):
                
                Q5-Q6: Literal retrieval about the narrative (2 questions, 2 marks each)
                  - "Based on Paragraph X, what TWO things did the narrator notice when...?"
                  - "According to the passage, why did [character] decide to...?"
                  - "What happened after [event]?"
                  - "State two feelings the narrator experienced during..."
                
                Q7: Vocabulary-in-context (1 mark)
                  - "Give one word from Paragraph X that suggests [feeling/atmosphere/action]..."
                  - "What word in the passage conveys [meaning]...?"
                
                Q8: Writer's craft / Language effect (2 marks)
                  - NARRATIVE-FOCUSED PHRASING:
                  - "How does the writer create tension/suspense in Paragraph X?"
                  - "How does the phrase '[exact quote]' contribute to the mood of the passage?"
                  - "What is the effect of the writer's description of [scene/character]?"
                  - "How does the writer convey the character's emotions through [technique]?"
                
                Q9: Evaluative/Opinion about the narrative (2-3 marks)
                  - "What do you think the narrator learned from this experience? Explain with evidence."
                  - "Do you think [character's decision] was the right choice? Explain your view."
                  - "What does this story suggest about [theme]?"
                
                Q10: Sequence Flowchart (4 marks) - STORY SEQUENCE, not themes
                  - For NARRATIVE passages, the flowchart shows the SEQUENCE OF EVENTS
                  - Structure: What happened first → What happened next → Then → Finally
                  - Provide 6 options, students choose 4 that show correct order of events

                FINAL QUESTION - SEQUENCE FLOWCHART (Q10, worth 4 marks):

                *** FOR NARRATIVE: Track the SEQUENCE OF EVENTS across paragraphs ***

                FLOWCHART DESIGN FOR NARRATIVE:
                
                The flowchart should trace the story's plot sequence:
                - Box 1: Event/situation from Paragraph 1-2 (beginning)
                - Box 2: Event/development from Paragraph 2-3 (rising action)
                - Box 3: Event/climax from Paragraph 3-4 (middle/turning point)
                - Box 4: Event/resolution from Paragraph 4-5 (end)
                
                FLOWCHART OPTIONS - Create 6 event descriptions:
                - 4 CORRECT options: Each summarizes a key event from a specific part of the story
                - 2 DISTRACTOR options: Events that did NOT happen or happen in wrong order
                
                EXAMPLE FOR NARRATIVE:

                Story about a student's first day at a new school:
                - Para 1-2: Nervous arrival, looking at unfamiliar faces
                - Para 2-3: First class, struggles to find the classroom
                - Para 3-4: Makes an unexpected friend during lunch
                - Para 4-5: Realizes the new school might not be so bad

                Options (showing story sequence):
                A. The narrator felt anxious while entering the school gates (→ Beginning)
                B. The narrator immediately felt welcomed by everyone (DISTRACTOR - not what happened)
                C. The narrator got lost trying to find the first classroom (→ Rising action)
                D. A classmate invited the narrator to sit together at lunch (→ Middle)
                E. The narrator decided to transfer to another school (DISTRACTOR - not what happened)
                F. The narrator felt hopeful about the days ahead (→ Resolution)

                ===FLOWCHART_ANSWER_KEY_START===
                Box 1 (Beginning): A (reason: describes initial nervousness in Para 1-2)
                Box 2 (Rising action): C (reason: describes getting lost in Para 2-3)
                Box 3 (Middle): D (reason: describes making friend in Para 3-4)
                Box 4 (Resolution): F (reason: describes final positive outlook in Para 4-5)
                Distractors: B, E (B: contradicts initial feelings; E: not stated in passage)
                ===FLOWCHART_ANSWER_KEY_END===
                """
            ),
            "section_c": dedent(
                """\
                Generate Paper 2 Section C [25 marks] (Guided Comprehension + Summary), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section C content.
                - Start directly with "**Section C [25 marks]**" or "Section C [25 marks]".
                
                *** OFFICIAL SYLLABUS REQUIREMENT: Text 4 MUST be NON-NARRATIVE in nature ***
                
                - Provide a NON-NARRATIVE passage (~550–650 words) with paragraph numbers
                - NON-NARRATIVE means: expository, argumentative, informational, explanatory, persuasive
                - NOT a story, NOT a personal recount, NOT fiction
                - Section B is NARRATIVE (story), so Section C must be NON-NARRATIVE (factual/informational)
                - Do NOT reuse Section B's theme

                QUESTION NUMBERING - CRITICAL:
                - Section C continues from Section B (which ends at Q10)
                - Section C questions are: Q11, Q12, Q13, Q14 (comprehension) and Q15 (summary)
                - Do NOT restart at Q1

                COMPREHENSION QUESTIONS (Q11-Q14, 10 marks total):
                
                ALLOWED QUESTION TYPES:
                1. Literal Retrieval [1-2 marks]
                   - "What are two benefits of...?" 
                   - "According to Paragraph X, what/why/how...?"
                   - "State one reason..."

                2. Explanation/Reasoning [2 marks]
                   - "How does the writer show that X is important in Paragraph Y?"
                   - "Explain how X contributes to Y according to the passage."
                   - CLEAR PHRASING - avoid vague "influence" questions

                3. Vocabulary-in-Context [1 mark]
                   - "Give one word from Paragraph X that means '...'."

                4. Cause-Effect [2 marks]
                   - "What effect does X have on Y?"

                DO NOT INCLUDE:
                ✗ "Based on your own knowledge..."
                ✗ "Do you agree...? Justify with your own views"
                ✗ Open-ended evaluative questions
                ✗ Writer's tone/attitude analysis

                SUMMARY TASK (Q15, 15 marks):
                
                PASSAGE DESIGN FOR SUMMARY:
                - The summary paragraphs MUST contain AT LEAST 10 DISTINCT summarisable points
                - Each point should be a separate fact, benefit, feature, or idea
                - Points should be clearly identifiable (not buried in complex sentences)

                PARAGRAPH RANGE FOR SUMMARY:
                - Use a SUBSET of paragraphs, NOT all paragraphs
                - CORRECT: "Paragraphs 2-5" or "Paragraphs 2-4" (specific body paragraphs)
                - AVOID: "Paragraphs 1-6" (too broad, includes intro and conclusion)
                - The subset should contain the main content paragraphs with summarisable points
                - Introduction (Para 1) and Conclusion (final para) typically excluded

                FORMAT:
                "15. Summary Task [15 marks]

                Using your own words as far as possible, summarise [specific focus] as described in Paragraphs X-Y.

                Your summary must be in continuous writing (not note form) and should not be longer than 80 words, including the 10 words given below to help you begin.

                [Starting line with approximately 10 words]..."

                EXAMPLE:
                "Using your own words as far as possible, summarise the benefits of civic engagement as described in Paragraphs 2-5.
                
                Your summary must be in continuous writing (not note form) and should not be longer than 80 words, including the 10 words given below to help you begin.
                
                Civic engagement benefits communities and individuals in several ways..."

                MARK BREAKDOWN (internal, not shown to students):
                - Content: 8 marks (1 mark per key point, max 8)
                - Language: 7 marks (paraphrasing quality, fluency, coherence)

                - Ensure total marks are exactly 25 (10 for Q11-Q14 + 15 for Q15).
                """
            ),
        },
        "oral": {
            None: dedent(
                """\
                Generate a complete GCE O-Level Oral Communication examination with ALL THREE components:

                COMPONENT 1: READING ALOUD [10 marks]
                - Provide ONE prose passage of 300-400 words
                - Topic should be contemporary and engaging (technology, environment, social issues, culture)
                - Include a mix of sentence structures: simple, compound, and complex
                - Include dialogue or direct speech (1-2 instances)
                - Include numbers, dates, or statistics that require clear articulation
                - Include words with varied stress patterns and challenging pronunciations
                - Mark the passage with suggested pause points using "/" for short pauses and "//" for longer pauses
                - Note any challenging words in brackets with pronunciation guidance

                COMPONENT 2: STIMULUS-BASED CONVERSATION (SBC) [20 marks]
                - Provide a VISUAL STIMULUS description (poster, infographic, or advertisement)
                - The visual should relate to a contemporary issue or topic
                - Include key visual elements: headings, statistics, images described, callouts
                - After the visual, provide 4 DISCUSSION PROMPTS:
                  • Q1: Direct reference to stimulus content (literal/factual)
                  • Q2: Personal opinion/experience related to stimulus theme
                  • Q3: Broader implications/analysis of the issue
                  • Q4: Hypothetical scenario or solution-based question
                - Include examiner notes with potential follow-up probes

                COMPONENT 3: GENERAL CONVERSATION [20 marks]
                - Provide 5 CONVERSATION THEMES, each with:
                  • Theme title (e.g., "Technology & Daily Life")
                  • 3-4 guiding questions per theme
                  • Questions should progress: factual → personal → analytical → evaluative
                - Themes should cover diverse areas: personal, social, educational, global
                - Include examiner guidance on follow-up questions

                FORMAT REQUIREMENTS:
                - Clear section headers for each component
                - Timing guidance (Reading: 10 min prep, 2 min read; SBC: 10 min; Conversation: 10 min)
                - Candidate instructions at the start of each section
                """
            ),
            "reading_aloud": dedent(
                """\
                Generate ONLY the READING ALOUD component [10 marks] of the Oral Examination:

                PASSAGE REQUIREMENTS:
                - Length: 300-400 words of continuous prose
                - Topic: Contemporary and relevant (technology, environment, health, social issues, culture, travel)
                - Tone: Narrative, descriptive, or expository (NOT argumentative for reading aloud)

                LINGUISTIC FEATURES TO INCLUDE:
                - Variety of sentence lengths and structures
                - 1-2 instances of direct speech or dialogue
                - Numbers, dates, percentages, or statistics (e.g., "67 percent", "2.5 million", "1997")
                - Proper nouns and place names requiring clear pronunciation
                - Words with challenging stress patterns or pronunciations
                - Emotive or descriptive vocabulary
                - Connectives and transitional phrases

                FORMAT:
                - Start with "READING ALOUD [10 marks]" header
                - Include timing: "Preparation time: 10 minutes | Reading time: Approximately 2 minutes"
                - Title the passage appropriately
                - Present the passage in clear paragraphs
                - After the passage, include:
                  • "Pronunciation Guide:" section with 3-5 challenging words and their phonetic hints
                  • "Suggested Pause Points:" brief guidance on natural pausing

                DIFFICULTY CALIBRATION:
                - Foundational: Simpler vocabulary, shorter sentences, familiar topics
                - Standard: Balanced complexity, varied structures, contemporary topics
                - Advanced: Sophisticated vocabulary, complex structures, nuanced topics
                """
            ),
            "sbc": dedent(
                """\
                Generate ONLY the STIMULUS-BASED CONVERSATION (SBC) component [20 marks]:

                VISUAL STIMULUS REQUIREMENTS:
                - Describe a visual stimulus in detail (poster, infographic, advertisement, or webpage)
                - Topic: Contemporary issue relevant to students (social media, environment, education, health, technology)
                - Include these elements in your description:
                  • Main heading/title
                  • Key statistics or facts (at least 2-3)
                  • Visual elements (images, icons, graphics - describe what they show)
                  • Callout boxes or highlighted information
                  • Any slogans, taglines, or quotes
                  • Organization/source attribution

                DISCUSSION PROMPTS (4 questions):

                Question 1 - Stimulus-Based (Factual):
                - Direct reference to information in the visual
                - E.g., "According to the infographic, what is the main cause of...?"
                - Should be answerable from the stimulus content

                Question 2 - Personal Response:
                - Connects stimulus theme to candidate's experience/opinion
                - E.g., "How do you personally feel about...?" or "In your experience, have you...?"

                Question 3 - Analysis/Implications:
                - Broader thinking about the issue
                - E.g., "Why do you think this is becoming more common...?" or "What are the consequences of...?"

                Question 4 - Hypothetical/Solution:
                - Forward-thinking or problem-solving
                - E.g., "If you were in charge of..., what would you do?" or "How might we address...?"

                EXAMINER NOTES:
                - Include 2-3 potential follow-up probes for each question
                - Note areas to explore if candidate gives brief responses

                FORMAT:
                - Start with "STIMULUS-BASED CONVERSATION [20 marks]" header
                - Include timing: "Discussion time: Approximately 10 minutes"
                - Present visual stimulus description in a bordered/highlighted section
                - Number questions clearly as Q1, Q2, Q3, Q4
                """
            ),
            "conversation": dedent(
                """\
                Generate ONLY the GENERAL CONVERSATION component [20 marks]:

                THEME REQUIREMENTS:
                - Provide EXACTLY 5 conversation themes
                - Themes should be diverse and age-appropriate for O-Level students (15-17 years)
                - Each theme should allow for personal, analytical, and evaluative responses

                SUGGESTED THEME CATEGORIES (choose 5):
                1. Personal & Family: relationships, responsibilities, values
                2. School & Education: learning, teachers, future plans
                3. Friends & Social Life: friendships, peer pressure, socializing
                4. Technology & Media: social media, gaming, digital life
                5. Environment & Society: sustainability, community, social issues
                6. Health & Lifestyle: wellbeing, sports, habits
                7. Culture & Traditions: festivals, customs, identity
                8. Future & Aspirations: career, goals, dreams
                9. Travel & Experiences: places, adventures, memories
                10. Current Affairs: news, global issues, local matters

                FOR EACH THEME, PROVIDE:

                Theme Title: [Clear, engaging title]

                Questions (3-4 per theme, progressing in complexity):
                1. Factual/Personal: Simple question about candidate's experience
                   E.g., "Tell me about your family" or "What do you enjoy doing in your free time?"

                2. Descriptive/Explanatory: Requires more detail
                   E.g., "Describe a memorable experience..." or "Explain why you feel..."

                3. Analytical: Requires reasoning or comparison
                   E.g., "Why do you think young people...?" or "How has this changed over time?"

                4. Evaluative/Hypothetical: Requires judgement or speculation
                   E.g., "What would you do if...?" or "Do you think this is a good development?"

                EXAMINER GUIDANCE:
                - For each theme, include 2-3 follow-up prompts
                - Note how to encourage elaboration if responses are brief
                - Suggest areas to probe for more depth

                FORMAT:
                - Start with "GENERAL CONVERSATION [20 marks]" header
                - Include timing: "Conversation time: Approximately 10 minutes"
                - Use "THEME 1:", "THEME 2:", etc. as headers
                - Number questions within each theme
                - Place examiner notes in italics or brackets
                """
            ),
        },
    }

    guidance_map = base_guidance.get(paper_format, {})
    section_key = section if section in guidance_map else None
    guidance = guidance_map.get(section_key)
    if not guidance:
        return "Follow the standard format for the selected paper."
    return guidance.strip()


def _load_reference_excerpt(paper_format: str, max_chars: int = 1000) -> Optional[str]:
    """
    Load a small excerpt from an existing parsed paper to guide tone/structure.
    Chooses a Paper 1 or Paper 2 reference based on the requested format.
    """
    try:
        texts_dir = settings.ocr_output_dir
        if not texts_dir.exists():
            return None
        # Heuristic filename filters
        target_key = "Paper-1" if paper_format == "paper_1" else "Paper-2"
        candidates: List[Path] = []
        for path in texts_dir.rglob("*.txt"):
            name = path.name
            if target_key in name:
                candidates.append(path)
        if not candidates:
            # fallback: any .txt
            candidates = list(texts_dir.rglob("*.txt"))
        if not candidates:
            return None
        # pick the newest by modified time
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ref = candidates[0]
        content = ref.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return None
        return content[:max_chars]
    except Exception:
        return None


def _ensure_openai_client(client: Optional[OpenAI] = None) -> OpenAI:
    if client is not None:
        return client

    api_key = settings.openai_api_key
    if not api_key:
        raise PaperGenerationError(
            "OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable."
        )

    return OpenAI(api_key=api_key)


def _render_pdf(text: str, output_path: Path, *, paper_format: Optional[str] = None, section: Optional[str] = None) -> None:
    """
    Render text to PDF using ReportLab Platypus (A4, styled paragraphs, lists) to avoid overflow
    and approximate official exam layout more closely.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Document setup: A4 with comfortable margins
    left_margin = right_margin = 20 * mm
    top_margin = 20 * mm
    bottom_margin = 20 * mm
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="GCE English Paper",
        author="GCE English Backend",
    )

    # Styles
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "Base",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "Heading1",
        parent=base,
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=6,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "Heading2",
        parent=base,
        fontName="Times-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=6,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=base,
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=8,
    )

    def _to_paragraph(text_line: str, style: ParagraphStyle) -> Paragraph:
        # Minimal inline bold: convert **bold** to <b>bold</b>
        # This is a simple pass; avoids unmatched markers
        line = text_line
        parts = line.split("**")
        if len(parts) >= 3 and len(parts) % 2 == 1:
            rebuilt = []
            for i, p in enumerate(parts):
                if i % 2 == 1:
                    rebuilt.append(f"<b>{p}</b>")
                else:
                    rebuilt.append(p)
            line = "".join(rebuilt)
        return Paragraph(line, style)

    story: List[object] = []

    # Optional header scaffold to approximate official look
    if paper_format in {"paper_1", "paper_2"}:
        code = "1128/01" if paper_format == "paper_1" else "1128/02"
        title = "Paper 1 Writing" if paper_format == "paper_1" else "Paper 2 Comprehension"
        story.append(_to_paragraph("MINISTRY OF EDUCATION, SINGAPORE", h2))
        story.append(_to_paragraph("in collaboration with", base))
        story.append(_to_paragraph("UNIVERSITY OF CAMBRIDGE LOCAL EXAMINATIONS SYNDICATE", h2))
        story.append(Spacer(1, 6))
        story.append(_to_paragraph("General Certificate of Education Ordinary Level", base))
        story.append(_to_paragraph("ENGLISH LANGUAGE", h2))
        story.append(_to_paragraph(code, base))
        story.append(_to_paragraph(title, base))
        story.append(Spacer(1, 10))

    # Specialized formatting for P1 Section A numbered 12-line passage + answer spaces
    def _try_render_p1_section_a(lines: List[str]) -> Optional[List[object]]:
        if not (paper_format == "paper_1" and (section == "section_a" or "Section A" in text or "Section A" in "\n".join(lines[:5]))):
            return None
        # Extract 12 numbered lines in the form "1. text"
        numbered = []
        for raw in lines:
            s = raw.strip()
            if len(s) >= 3 and s[0].isdigit() and s[1] == "." and s[2] == " ":
                try:
                    num = int(s.split(".", 1)[0])
                except ValueError:
                    continue
                content = s.split(".", 1)[1].strip()
                numbered.append((num, content))
        nums = [n for n, _ in numbered]
        if not (len(numbered) >= 12 and set(range(1, 13)).issubset(set(nums))):
            return None
        # Build a table for the 12 lines
        data = [["Line", "Text"]]
        for idx in range(1, 13):
            content = next((c for n, c in numbered if n == idx), "")
            data.append([str(idx), content])
        tbl = Table(data, colWidths=[20 * mm, doc.width - 20 * mm])
        tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 1), (-1, -1), 11),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ALIGN", (0, 1), (0, -1), "RIGHT"),
                ]
            )
        )
        output: List[object] = []
        output.append(_to_paragraph("Section A [10 marks] (Editing)", section_style))
        output.append(tbl)
        output.append(Spacer(1, 8))
        # Answer spaces
        answer_rows = []
        for i in range(1, 13):
            answer_rows.append([f"{i}.", "_" * 80])
        ans_tbl = Table(answer_rows, colWidths=[10 * mm, doc.width - 10 * mm])
        ans_tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        output.append(_to_paragraph("Answer Spaces:", base))
        output.append(ans_tbl)
        return output

    # Parse lines into flowables with simple rules
    lines = text.splitlines()

    # If this looks like P1 Section A, try the exact table layout
    p1a = _try_render_p1_section_a(lines)
    if p1a:
        story.extend(p1a)
        doc.build(story)
        return
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        if not raw.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Headings via markdown-like markers
        if raw.startswith("# "):
            story.append(_to_paragraph(raw[2:].strip(), h1))
            i += 1
            continue
        if raw.startswith("## "):
            story.append(_to_paragraph(raw[3:].strip(), h2))
            i += 1
            continue
        # Section labels
        low = raw.lower()
        if low.startswith("section a") or low.startswith("section b") or low.startswith("section c"):
            story.append(_to_paragraph(raw.strip(), section_style))
            i += 1
            continue

        # Bulleted list block
        if raw.lstrip().startswith(("- ", "* ")):
            items: List[ListItem] = []
            while i < len(lines):
                line = lines[i].rstrip()
                if line.lstrip().startswith(("- ", "* ")):
                    content = line.lstrip()[2:].strip()
                    items.append(ListItem(_to_paragraph(content, base)))
                    i += 1
                else:
                    break
            story.append(ListFlowable(items, bulletType="bullet", bulletFontName="Times-Roman"))
            continue

        # Numbered list block like "1. "
        stripped = raw.lstrip()
        if len(stripped) > 3 and stripped[0].isdigit() and stripped[1] in {".", ")"} and stripped[2] == " ":
            items: List[ListItem] = []
            while i < len(lines):
                cand = lines[i].lstrip()
                if len(cand) > 3 and cand[0].isdigit() and cand[1] in {".", ")"} and cand[2] == " ":
                    items.append(ListItem(_to_paragraph(cand[3:].strip(), base)))
                    i += 1
                else:
                    break
            story.append(ListFlowable(items, bulletType="1"))
            continue

        # Default paragraph
        story.append(_to_paragraph(raw, base))
        i += 1

    doc.build(story)


def _render_html_then_pdf(
    *,
    content: str,
    pdf_path: Path,
    html_path: Path,
    paper_format: Optional[str],
    section: Optional[str],
    visual_image_path: Optional[Path] = None,
    visual_caption: Optional[str] = None,
) -> None:
    # Defaults for header meta if not provided elsewhere
    default_session = "October/November"
    default_year = str(datetime.utcnow().year)
    default_duration = "1 hour 50 minutes" if paper_format in {"paper_1", "paper_2"} else None
    watermark = "MINISTRY OF EDUCATION, SINGAPORE"
    render_html_template(
        paper_format=paper_format or "",
        section=section,
        content=content,
        output_html=html_path,
        session=default_session,
        year=default_year,
        duration=default_duration,
        watermark_text=watermark,
        visual_image_path=visual_image_path,
        visual_caption=visual_caption,
    )
    html_to_pdf(html_path, pdf_path)


def generate_paper(
    *,
    difficulty: str,
    paper_format: str,
    section: Optional[str] = None,
    topics: Optional[Iterable[str]] = None,
    additional_instructions: Optional[str] = None,
    client: Optional[OpenAI] = None,
    visual_mode: str = "embed",
    search_provider: str = "hybrid",
    user_id: Optional[str] = None,
    generate_answer_key_flag: bool = False,
) -> PaperGenerationResult:
    """Generate a synthetic exam paper using the configured LLM."""

    llm_client = _ensure_openai_client(client)

    # Auto-visuals for P1 Section B, P2 Section A, and Oral (Stimulus-Based Conversation)
    snapshot: Optional[VisualSnapshot] = None
    visual_description: Optional[str] = None
    wants_visuals = visual_mode in {"embed", "auto"}
    # Fetch visuals for visual sections; for full papers (section is None), fetch for the canonical visual section
    needs_visuals = (
        (paper_format == "paper_1" and (section in (None, "section_b"))) or
        (paper_format == "paper_2" and (section in (None, "section_a"))) or
        (paper_format == "oral")  # Oral exams need visuals for Stimulus-Based Conversation
    )
    if wants_visuals and needs_visuals:
        try:
            snapshot, visual_description = get_visual(
                topics=topics,
                paper_format=paper_format,
                section=section,
                search_provider=search_provider,
            )
        except Exception:
            snapshot, visual_description = None, None

    if snapshot:
        logger.info(
            "Visual selected",
            url=snapshot.url,
            title=snapshot.title,
            host=snapshot.host,
            shot_exists=bool(snapshot.screenshot_path and snapshot.screenshot_path.exists()),
        )

    # Helper to generate one section at a time with validation and retry
    def _gen_one(sec: Optional[str], extra_visual_desc: Optional[str] = None, max_retries: int = 2) -> Tuple[str, str]:
        pr = _build_prompt(
            difficulty=difficulty,
            paper_format=paper_format,
            section=sec,
            topics=topics,
            additional_instructions=additional_instructions,
        )
        if extra_visual_desc:
            pr += "\n\nVisual stimulus (text description):\n" + extra_visual_desc.strip()
            pr += (
                "\nIMPORTANT: Even if the visual description contains non-English text, "
                "ALL generated content (questions, instructions, prompts) must be EXCLUSIVELY in English. "
                "A visual stimulus image will be embedded in the paper alongside this question. "
                "You MUST acknowledge this visual stimulus in your task instructions by explicitly referencing it "
                "(e.g., 'Refer to the visual stimulus provided above' or 'Using the visual stimulus shown' or "
                "'You have come across a visually appealing webpage/poster...'). "
                "Do NOT restate or copy the visual description verbatim in the paper output. "
                "Instead, write task instructions that reference the visual and guide students on what to write based on it. "
                "Write only the task instructions/questions and marking guidance as required, all in English."
            )

        # Enhance prompt with RAG context from past papers
        pr = get_rag_enhanced_prompt(
            pr,
            paper_format=paper_format,
            section=sec,
            topics=list(topics) if topics else None,
            difficulty=difficulty,
            max_context_chunks=5,  # Increased for better context
        )

        # Get section-specific temperature
        section_temp = _get_section_temperature(paper_format, sec)

        logger.info(
            "Requesting LLM generated paper",
            difficulty=difficulty,
            paper_format=paper_format,
            section=sec,
            topics=list(topics) if topics else None,
            temperature=section_temp,
        )

        last_error: Optional[Exception] = None
        last_issues: List[str] = []

        for attempt in range(max_retries + 1):
            try:
                # Build params with section-specific temperature
                params = {**LLM_COMPLETION_PARAMS, "temperature": section_temp}

                # On retry, slightly increase temperature and add feedback
                retry_prompt = pr
                if attempt > 0:
                    params["temperature"] = min(section_temp + 0.1 * attempt, 0.8)
                    retry_prompt = pr + f"\n\nPREVIOUS ATTEMPT HAD ISSUES: {'; '.join(last_issues)}. Please fix these issues in this attempt."
                    logger.info(f"Retry attempt {attempt} with adjusted temperature {params['temperature']}")

                response_local = llm_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert curriculum designer creating official-style English "
                                "examination papers. All content must be written EXCLUSIVELY in English. "
                                "Never generate content in any other language, including Urdu, Arabic, or any non-English language. "
                                "Follow the exact format and structure requirements precisely."
                            ),
                        },
                        {"role": "user", "content": retry_prompt},
                    ],
                    **params,
                )

                out = response_local.choices[0].message.content.strip()
                if not out:
                    raise PaperGenerationError("LLM returned empty content for the paper")

                # Validate content
                is_valid, validation_issues = _validate_content(out, paper_format, sec)
                llm_issues = _check_common_llm_issues(out)

                all_issues = validation_issues + llm_issues

                if all_issues:
                    logger.warning(f"Content issues found (attempt {attempt + 1}): {all_issues}")
                    last_issues = all_issues

                    # Only retry if there are serious issues and we have retries left
                    serious_issues = [i for i in all_issues if "too short" in i.lower() or "too few" in i.lower() or "non-english" in i.lower()]
                    if serious_issues and attempt < max_retries:
                        continue  # Try again

                logger.info(f"LLM content generated | chars={len(out)} | attempt={attempt + 1}")
                logger.info(f"LLM raw response preview: {out[:2000]}{'...[truncated]' if len(out) > 2000 else ''}")

                return out, pr

            except PaperGenerationError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {exc}")
                    continue
                raise PaperGenerationError(f"Failed to generate paper via LLM after {max_retries + 1} attempts") from exc

        # Should not reach here, but just in case
        if last_error:
            raise PaperGenerationError("Failed to generate paper via LLM") from last_error
        raise PaperGenerationError("Failed to generate paper via LLM")

    # Generate content: single section or full paper in three calls
    combined_prompts: List[str] = []
    section_a_error_key: Optional[Dict[str, any]] = None  # Track Section A error key
    flowchart_answer_key: Optional[Dict[str, any]] = None  # Track flowchart answer key for Paper 2 Section B

    if section is not None:
        extra = visual_description if (visual_description and ((paper_format == "paper_1" and section == "section_b") or (paper_format == "paper_2" and section == "section_a"))) else None
        content, pr_used = _gen_one(section, extra)
        combined_prompts.append(pr_used)

        # Extract error key for Section A
        if paper_format == "paper_1" and section == "section_a":
            content, section_a_error_key = _extract_section_a_error_key(content)
        # Extract flowchart answer key for Paper 2 Section B
        if paper_format == "paper_2" and section == "section_b":
            content, flowchart_answer_key = _extract_flowchart_answer_key(content)
    else:
        if paper_format == "paper_1":
            a, pr_a = _gen_one("section_a")
            # Extract error key from Section A before combining
            a, section_a_error_key = _extract_section_a_error_key(a)
            b, pr_b = _gen_one("section_b", visual_description)
            c, pr_c = _gen_one("section_c")
            content = "\n\n".join([a, b, c])
            combined_prompts.extend([pr_a, pr_b, pr_c])
        elif paper_format == "paper_2":
            a, pr_a = _gen_one("section_a", visual_description)
            b, pr_b = _gen_one("section_b")
            # Extract flowchart answer key from Section B before combining
            b, flowchart_answer_key = _extract_flowchart_answer_key(b)
            c, pr_c = _gen_one("section_c")
            content = "\n\n".join([a, b, c])
            combined_prompts.extend([pr_a, pr_b, pr_c])
        elif paper_format == "oral":
            # Generate all three oral components
            # Visual stimulus is used for the Stimulus-Based Conversation (SBC) section
            reading, pr_r = _gen_one("reading_aloud")
            sbc, pr_s = _gen_one("sbc", visual_description)  # Pass visual description to SBC
            conv, pr_c = _gen_one("conversation")
            content = "\n\n---\n\n".join([reading, sbc, conv])
            combined_prompts.extend([pr_r, pr_s, pr_c])
        else:
            # Fallback single prompt
            content, pr_single = _gen_one(None)
            combined_prompts.append(pr_single)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_format = "".join(ch for ch in paper_format if ch.isalnum() or ch in {"-", "_"}).lower()
    safe_difficulty = "".join(ch for ch in difficulty if ch.isalnum() or ch in {"-", "_"}).lower()
    safe_section = ""
    if section:
        safe_section = "".join(ch for ch in section if ch.isalnum() or ch in {"-", "_"})  # already lower
        safe_section = f"-{safe_section.lower()}"

    base_name = f"{safe_format}{safe_section}-{safe_difficulty}-{timestamp}"
    pdf_path = settings.paper_output_dir / f"{base_name}.pdf"
    text_path = settings.paper_output_dir / f"{base_name}.txt"
    html_path = settings.paper_output_dir / f"{base_name}.html"

    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(content, encoding="utf-8")
    # Produce both HTML (for visual inspection/template tweaking) and PDF (for distribution)
    # Compute relative visual path if any
    visual_image_rel: Optional[Path] = None
    visual_caption: Optional[str] = None
    if snapshot and snapshot.screenshot_path and snapshot.screenshot_path.exists() and wants_visuals:
        try:
            rel = os.path.relpath(snapshot.screenshot_path, start=html_path.parent)
            visual_image_rel = Path(rel)
            visual_caption = f"{snapshot.title} — {snapshot.host}" if snapshot.host else snapshot.title
        except Exception:
            visual_image_rel = snapshot.screenshot_path
            visual_caption = snapshot.title

    _render_html_then_pdf(
        content=content,
        pdf_path=pdf_path,
        html_path=html_path,
        paper_format=paper_format,
        section=section,
        visual_image_path=visual_image_rel,
        visual_caption=visual_caption,
    )

    # Upload to Supabase Storage in a per-user, per-paper-type path if user_id is known
    download_url: Optional[str] = None
    try:
        storage_key = f"{base_name}.pdf"
        if user_id:
            safe_user = "".join(ch for ch in user_id if ch.isalnum() or ch in {"-", "_"})
            paper_folder = "paper1" if paper_format == "paper_1" else "paper2" if paper_format == "paper_2" else paper_format
            storage_key = f"{safe_user}/{paper_folder}/{base_name}.pdf"

        download_url = upload_generated_paper_pdf(pdf_path, object_key=storage_key)
    except SupabaseError as exc:
        # Log but don't fail the whole generation if storage is misconfigured
        logger.error(f"Failed to upload generated paper to Supabase Storage: {exc}")

    logger.info(
        "Generated synthetic paper",
        pdf=str(pdf_path),
        text=str(text_path),
        difficulty=difficulty,
        paper_format=paper_format,
        section=section,
        visual_embedded=bool(visual_image_rel),
    )

    # Generate answer key if requested
    answer_key_data: Optional[Dict[str, any]] = None
    answer_key_pdf: Optional[Path] = None

    if generate_answer_key_flag:
        try:
            logger.info("Generating answer key...")
            answer_key_result = generate_answer_key(
                paper_content=content,
                paper_format=paper_format,
                section=section,
                client=llm_client,
                section_a_error_key=section_a_error_key,  # Pass pre-extracted error key
            )
            answer_key_data = answer_key_result.to_dict()

            # Save answer key JSON
            answer_key_json_path = settings.paper_output_dir / f"{base_name}-answer-key.json"
            save_answer_key_json(answer_key_result, answer_key_json_path)

            # Render answer key PDF
            answer_key_pdf = settings.paper_output_dir / f"{base_name}-answer-key.pdf"
            render_answer_key_pdf(answer_key_result, answer_key_pdf)

            logger.info(f"Answer key generated: {answer_key_pdf}")
        except AnswerKeyError as exc:
            logger.error(f"Failed to generate answer key: {exc}")
            # Don't fail the whole generation, just skip the answer key

    return PaperGenerationResult(
        content=content,
        prompt="\n\n---\n\n".join(combined_prompts),
        pdf_path=pdf_path,
        text_path=text_path,
        created_at=datetime.utcnow(),
        section=section,
        visual_meta=(
            {
                "url": snapshot.url,
                "title": snapshot.title,
                "host": snapshot.host,
            }
            if snapshot
            else None
        ),
        download_url=download_url,
        answer_key=answer_key_data,
        answer_key_pdf_path=answer_key_pdf,
        section_a_error_key=section_a_error_key,  # Include extracted error key
    )


