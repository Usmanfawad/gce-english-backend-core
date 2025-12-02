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
from app.services.visuals import get_visual, VisualSnapshot
from app.services.html_renderer import html_to_pdf, render_html_template
from app.services.rag import get_rag_enhanced_prompt


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


LLM_COMPLETION_PARAMS = {
    "temperature": 0.35,
    "top_p": 0.9,
    "frequency_penalty": 0.15,
    "presence_penalty": 0.0,
    "max_tokens": 6000,
}


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
                Generate Paper 1 Section A [10 marks] (Editing), SINGLE SECTION ONLY:
                
                PASSAGE FORMAT:
                - Output ONE passage formatted into exactly 12 lines, each line prefixed "1. ", "2. ", ..., "12. ".
                - CRITICAL: The passage MUST be CONTINUOUS PROSE—a single flowing paragraph that has been divided into numbered lines for editing purposes. 
                - Each line should flow naturally into the next, NOT be a standalone sentence. The text should read as one coherent narrative/description when the line numbers are removed.
                - WRONG: "1. The weather was nice. 2. I went to the park. 3. Birds were singing." (disconnected sentences)
                - CORRECT: "1. The weather was particularly pleasant that morning, with a gentle breeze 2. carrying the scent of freshly cut grass across the park where I had 3. decided to spend my afternoon reading under the old oak tree that..." (continuous flow)
                
                ERROR PLACEMENT:
                - Lines 1 and 12 must remain error-free; NEVER insert errors into those two lines.
                - Among lines 2–11, plant EXACTLY EIGHT single-word or single-phrase errors (grammar, vocabulary, spelling, or punctuation) and leave EXACTLY TWO of those lines completely correct.
                - Distribute the 8 errors unpredictably across lines 2–11 (not consecutive).
                - Do NOT highlight, flag, or hint where the errors are.
                
                ANSWER SPACES (DO NOT INCLUDE IN OUTPUT):
                - Do NOT include "Answer Spaces:" or any blank lines for answers. The HTML template will automatically generate the answer column on the right side of the passage.
                
                VALIDATION CHECKLIST (verify before returning):
                ✓ Exactly 12 numbered lines
                ✓ Continuous prose that flows naturally across all lines
                ✓ Lines 1 and 12 are error-free
                ✓ Exactly 8 errors distributed across lines 2–11
                ✓ Exactly 2 clean lines among lines 2–11
                ✓ No answer spaces or hints included
                """
            ),
            "section_b": dedent(
                """\
                Generate Paper 1 Section B [30 marks] (Situational Writing), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section B content.
                - Start directly with "**Section B [30 marks]**" or "Section B [30 marks]".
                
                VISUAL STIMULUS INTEGRATION:
                - If a visual stimulus is provided, you MUST acknowledge it in your task instructions. Begin with a statement like "You have come across a visually appealing webpage/poster/advertisement..." or "Refer to the visual stimulus provided..." or "Using the visual stimulus shown above..." to explicitly reference the image.
                - The visual should present COMPELLING, PERSUASIVE content that gives students clear reasons to choose between options or take action.
                - If the visual shows multiple options (e.g., courses, products, destinations), ensure the visual description includes DIFFERENTIATING FACTORS that help students make informed comparisons (e.g., price differences, unique benefits, target audiences, special features).
                
                TASK DESIGN:
                - Choose ONE appropriate task type (letter, email, report, or speech) consistent with the visual stimulus topic.
                - You MUST base the task on the provided visual description; reflect its headings/callouts/blurbs. Do NOT invent unrelated school or funding contexts unless topics explicitly suggest school.
                
                PAC (Purpose, Audience, Context) - IMPLICIT INTEGRATION:
                - DO NOT use explicit "Purpose:", "Audience:", "Context:" labels.
                - Instead, WEAVE the PAC elements naturally into the situational scenario. The purpose, audience, and context should be CLEAR from the narrative setup without being labeled.
                - WRONG: "Purpose: To persuade your friend. Audience: A close friend. Context: You saw an advertisement."
                - CORRECT: "Your close friend has been looking for a photography course to develop their hobby. You recently came across this advertisement and believe one of the courses would be perfect for them. Write an email to persuade them to sign up, explaining why you think it suits their interests and skill level."
                
                KEY POINTS & GUIDANCE:
                - List 3–5 key points students must address that directly tie to the visual description.
                - State tone/register explicitly (e.g., "Use an informal but enthusiastic tone").
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
                - Section B [20 marks] (Reading Comprehension Open-Ended):
                  Supply ONE nonfiction passage of about 600–650 words (coherent, contemporary topic), with paragraph numbers and LINE NUMBERS.
                  Prefix each line with a line number marker like "[1]", "[2]", ..., ensuring numbers increment correctly throughout the passage.
                  Set ~10–14 question parts (including sub-items) covering literal retrieval, inference, vocabulary-in-context,
                  writer's craft (phrased as "How does X persuade/influence the reader..."), and evaluation. Ensure at least ONE vocabulary-in-context item.
                  The FINAL question (Q10) MUST be a 4-mark flowchart with: paragraph-based labels (not Step 1-4), 6 options provided, and students choose 4 correct answers.
                - Section C [25 marks] (Guided Comprehension + Summary):
                  Provide a SECOND passage (different from Section B), around 400–550 words, with paragraph numbers and LINE NUMBERS.
                  Before the summary, set 4–5 questions totaling 10 marks (short-answer mix relevant to the new passage).
                  Then set a 15-mark summary task with a clearly specified focus and paragraph range; require continuous writing (≤80 words),
                  using own words as far as possible (not note form).
                """
            ),
            "section_a": dedent(
                """\
                Generate Paper 2 Section A [5 marks] (Visual Text Comprehension), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section A content.
                - Start directly with "**Section A [5 marks]**" or "Section A [5 marks]".
                - A visual stimulus image will be embedded in the paper. Begin your section by acknowledging it: "Refer to the visual stimulus provided above" or "Study the visual stimulus shown" or similar.
                - If a visual description is provided, use it to understand what the visual contains, but do NOT copy the description verbatim. Instead, write questions that reference specific elements that will be visible in the image.
                - Provide EXACTLY 4 questions where ONE has two subparts (e.g., Q1(a), Q1(b)) so the total marks is 5.
                - Q1(a) and Q1(b) must be literal retrieval questions requiring direct lifting of short phrases/words from the visual (no inference or opinion).
                - Q2 must analyse a specific persuasive element (e.g., the banner headline, a callout bubble, imagery, promises, emotional appeal) and require the student to name/explain the technique tied to that element.
                - Q3 must be a language effect question on a quoted word/phrase from the visual (e.g., "Explain the effect of the phrase '...')."
                - Q4 should be an inference question that still points back to explicit evidence in the visual.
                - Questions must refer directly to concrete elements of the visual stimulus (callouts, imagery, benefits, promises, emotional appeal, etc.).
                """
            ),
            "section_b": dedent(
                """\
                Generate Paper 2 Section B [20 marks] (Reading Comprehension Open-Ended), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section B content.
                - Start directly with "**Section B [20 marks]**" or "Section B [20 marks]".
                - Provide ONE nonfiction passage (~600–650 words) with clear paragraph numbering and LINE NUMBERS.
                - Prefix each line with a line number marker like "[1]", "[2]", ..., ensuring numbers increment correctly throughout the passage.
                - Set ~10–14 question parts (including sub-items) whose marks sum to EXACTLY 20. Clearly state the mark allocation for every part.
                
                REQUIRED QUESTION TYPES:
                  • Literal retrieval: 2–3 questions (direct lifting, short answers).
                  • Inferential: 2 questions.
                  • Vocabulary-in-context: 1–2 questions (at least one "Give one word/phrase..." style).
                  • Writer's craft / Language effect: 1 question analysing technique/effect.
                    - PHRASING: Use "How does [quoted phrase/technique] persuade the reader..." or "How does the writer influence the reader by using..." or "How does [X] affect the reader's perception of..."
                    - WRONG: "Explain the effect of the phrase '...'"
                    - CORRECT: "How does the phrase '...' persuade the reader to feel sympathy for the character?"
                    - CORRECT: "How does the writer's use of [technique] influence the reader's understanding of...?"
                  • Evaluative: 1–2 questions requiring judgement with textual evidence.
                  • Higher-order opinion/interpretation: 1 question inviting personal response linked to the passage.
                
                FINAL QUESTION - FLOWCHART (MUST be Q10, worth 4 marks):
                  • This MUST be the last question and presented as a flowchart/sequence task.
                  • Reference SPECIFIC PARAGRAPH RANGES (e.g., "Based on Paragraphs 2-3", "From Paragraphs 4-6").
                  • DO NOT use "Step 1, Step 2, Step 3, Step 4" format.
                  • USE paragraph-based labels like: "Paragraph 1-2:", "Paragraph 3-4:", "Paragraph 5:", "Paragraph 6-7:"
                  • Provide EXACTLY 6 OPTIONS for students to choose from, where only 4 are correct answers.
                  • Format example:
                    "10. Based on Paragraphs 2-6, complete the flowchart below by choosing FOUR correct answers from the six options provided. [4 marks]
                    
                    Options:
                    A. [statement about content from passage]
                    B. [statement about content from passage]
                    C. [statement about content from passage]
                    D. [statement about content from passage]
                    E. [statement about content from passage]
                    F. [statement about content from passage]
                    
                    Flowchart:
                    Paragraph 2-3: ______
                    Paragraph 4: ______
                    Paragraph 5: ______
                    Paragraph 6: ______"
                  
                - Reference paragraph or line numbers in the questions where helpful.
                """
            ),
            "section_c": dedent(
                """\
                Generate Paper 2 Section C [25 marks] (Guided Comprehension + Summary), SINGLE SECTION ONLY:
                - DO NOT include headers, footers, or paper metadata (MINISTRY OF EDUCATION, candidate info, etc.). Generate ONLY the Section C content.
                - Start directly with "**Section C [25 marks]**" or "Section C [25 marks]".
                - Provide a NEW nonfiction passage (~550–650 words), distinct from Section B, with paragraph numbers and LINE NUMBERS. Do NOT reuse Section B's passage or theme.
                - Set 4–5 short-answer questions totaling 10 marks. Include at least one inference question but avoid vocabulary-in-context and writer's effect/intention items for this section. Keep the difficulty comparable to Cambridge Paper 2 Section C.
                - Then set a 15-mark summary task with a clearly specified focus and paragraph range; require continuous writing (≤80 words) using candidates' own words as far as possible. Indicate that about 10 key points should be drawn from the passage.
                - Ensure total marks are exactly 25 (10 for questions + 15 for summary), and do not add extraneous headings.
                """
            ),
        },
        "oral": {
            None: dedent(
                """\
                Produce the three components of the Oral Communication paper: reading aloud passage, stimulus-based conversation prompt with background description, and general conversation follow-up themes.
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
) -> PaperGenerationResult:
    """Generate a synthetic exam paper using the configured LLM."""

    llm_client = _ensure_openai_client(client)

    # Auto-visuals for P1 Section B and P2 Section A
    snapshot: Optional[VisualSnapshot] = None
    visual_description: Optional[str] = None
    wants_visuals = visual_mode in {"embed", "auto"}
    # Fetch visuals for visual sections; for full papers (section is None), fetch for the canonical visual section
    needs_visuals = (paper_format == "paper_1" and (section in (None, "section_b"))) or (
        paper_format == "paper_2" and (section in (None, "section_a"))
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

    # Helper to generate one section at a time
    def _gen_one(sec: Optional[str], extra_visual_desc: Optional[str] = None) -> Tuple[str, str]:
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
            max_context_chunks=3,
        )
        
        logger.info(
            "Requesting LLM generated paper",
            difficulty=difficulty,
            paper_format=paper_format,
            section=sec,
            topics=list(topics) if topics else None,
        )
        try:
            response_local = llm_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert curriculum designer creating official-style English "
                            "examination papers. All content must be written EXCLUSIVELY in English. "
                            "Never generate content in any other language, including Urdu, Arabic, or any non-English language."
                        ),
                    },
                    {"role": "user", "content": pr},
                ],
                **LLM_COMPLETION_PARAMS,
            )
        except Exception as exc:
            raise PaperGenerationError("Failed to generate paper via LLM") from exc
        try:
            out = response_local.choices[0].message.content.strip()
        except Exception as exc:
            raise PaperGenerationError("Unexpected response format from LLM") from exc
        if not out:
            raise PaperGenerationError("LLM returned empty content for the paper")
        logger.info(f"LLM content generated | chars={len(out)}")
        logger.info(f"LLM raw response preview: {out[:2000]}{'...[truncated]' if len(out) > 2000 else ''}")
        return out, pr

    # Generate content: single section or full paper in three calls
    combined_prompts: List[str] = []
    if section is not None:
        extra = visual_description if (visual_description and ((paper_format == "paper_1" and section == "section_b") or (paper_format == "paper_2" and section == "section_a"))) else None
        content, pr_used = _gen_one(section, extra)
        combined_prompts.append(pr_used)
    else:
        if paper_format == "paper_1":
            a, pr_a = _gen_one("section_a")
            b, pr_b = _gen_one("section_b", visual_description)
            c, pr_c = _gen_one("section_c")
            content = "\n\n".join([a, b, c])
            combined_prompts.extend([pr_a, pr_b, pr_c])
        elif paper_format == "paper_2":
            a, pr_a = _gen_one("section_a", visual_description)
            b, pr_b = _gen_one("section_b")
            c, pr_c = _gen_one("section_c")
            content = "\n\n".join([a, b, c])
            combined_prompts.extend([pr_a, pr_b, pr_c])
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

    logger.info(
        "Generated synthetic paper",
        pdf=str(pdf_path),
        text=str(text_path),
        difficulty=difficulty,
        paper_format=paper_format,
        section=section,
        visual_embedded=bool(visual_image_rel),
    )

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
    )


