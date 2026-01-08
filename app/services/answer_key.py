"""Answer Key Generation Service for GCE English Papers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from app.config.settings import settings


class AnswerKeyError(RuntimeError):
    """Raised when answer key generation fails."""


@dataclass
class AnswerKey:
    """Generated answer key structure."""

    paper_format: str
    section: Optional[str]
    answers: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_format": self.paper_format,
            "section": self.section,
            "answers": self.answers,
            "created_at": self.created_at.isoformat(),
        }


# LLM parameters for answer key generation (more deterministic)
ANSWER_KEY_LLM_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 4000,
}


def _ensure_openai_client(client: Optional[OpenAI] = None) -> OpenAI:
    if client is not None:
        return client
    api_key = settings.openai_api_key
    if not api_key:
        raise AnswerKeyError("OpenAI API key not configured.")
    return OpenAI(api_key=api_key)


def _build_answer_key_prompt(paper_content: str, paper_format: str, section: Optional[str], section_a_error_key: Optional[Dict[str, Any]] = None) -> str:
    """Build the prompt for generating answer keys."""

    base_prompt = f"""You are an expert GCE O-Level English examiner. Analyze the following examination paper and generate a comprehensive answer key.

PAPER CONTENT:
{paper_content}

PAPER FORMAT: {paper_format}
SECTION: {section or "Full Paper"}

Generate answer keys in the following JSON structure. Be precise and follow official marking conventions.
"""

    if paper_format == "paper_1":
        if section == "section_a" or section is None:
            # If we have a pre-extracted error key, use it directly
            if section_a_error_key and section_a_error_key.get("errors"):
                errors_json = json.dumps(section_a_error_key["errors"], indent=8)
                correct_lines = section_a_error_key.get("correct_lines", [1, 12])
                base_prompt += f"""

FOR PAPER 1 SECTION A (EDITING):
The errors have been pre-identified during paper generation. Return this EXACT structure:
{{
    "section_a": {{
        "title": "Editing",
        "total_marks": 10,
        "errors": {errors_json},
        "correct_lines": {json.dumps(correct_lines)},
        "marking_notes": "Award 1 mark for each correct identification and correction. Each correction is a SINGLE WORD only."
    }}
}}

IMPORTANT: Use the pre-extracted errors EXACTLY as provided. Do NOT add or modify errors.
"""
            else:
                base_prompt += """

FOR PAPER 1 SECTION A (EDITING):

*** CRITICAL: Only mark lines as errors if they are CLEARLY grammatically wrong ***

Do NOT mark these as errors (they are correct):
- "all of the participants" - correct (NOT an error)
- "had accomplished" - correct (NOT an error)  
- "which were" - correct (NOT an error)
- "prepared the food" - correct (NOT an error)

Only mark CLEAR grammatical errors like:
- "students was" → were (subject-verb disagreement)
- "teacher have" → has (subject-verb disagreement)
- "Yesterday I walk" → walked (wrong tense)
- "ran very quick" → quickly (adjective instead of adverb)
- "recieved" → received (spelling)
- "better then" → than (word confusion)

Return JSON with this structure:
{
    "section_a": {
        "title": "Editing",
        "total_marks": 10,
        "errors": [
            {
                "line": 2,
                "error": "The single incorrect word",
                "correction": "The single correct word",
                "error_type": "subject-verb agreement/tense/spelling/word form/word choice",
                "explanation": "Brief explanation of why this is grammatically wrong"
            }
        ],
        "correct_lines": [1, 4, 8, 12],
        "marking_notes": "Award 1 mark for each correct identification. Each correction must be a SINGLE WORD."
    }
}

RULES:
- EXACTLY 8 errors in lines 2, 3, 5, 6, 7, 9, 10, 11
- EXACTLY 4 correct lines: 1, 4, 8, 12
- Each error = SINGLE WORD change only
- Do NOT invent errors in grammatically correct sentences
"""

        if section == "section_b" or section is None:
            base_prompt += """

FOR PAPER 1 SECTION B (SITUATIONAL WRITING):
Use the OFFICIAL GCE O-Level marking scheme: Task Fulfilment (10 marks) + Language (20 marks) = 30 marks

Return JSON with:
{
    "section_b": {
        "title": "Situational Writing",
        "total_marks": 30,
        "task_type": "email/letter/report/speech",
        "key_points": [
            {
                "point": "Description of required content point",
                "sample_response": "Example of how this point could be addressed"
            }
        ],
        "format_requirements": ["Opening salutation", "Closing", "Appropriate register"],
        "marking_rubric": {
            "task_fulfilment": {
                "total_marks": 10,
                "assessment_criteria": ["Addressing the required points", "Showing awareness of the purpose, audience and context", "Using the given information"],
                "bands": {
                    "Band 5 (9-10)": "All points addressed and developed in detail; Purpose, audience and context fully and clearly addressed; Ideas consistently supported by given information",
                    "Band 4 (7-8)": "All points addressed with one or more developed in detail; Purpose, audience and context clearly addressed; Ideas generally supported by given information",
                    "Band 3 (5-6)": "Most points addressed with some development; Purpose, audience and context addressed; Some attempts to use given information to support ideas",
                    "Band 2 (3-4)": "Some points addressed; Purpose, audience and context partially addressed; Some reference to given information",
                    "Band 1 (1-2)": "One point addressed; Purpose, audience and context occasionally addressed; Occasional reference to given information",
                    "Band 0 (0)": "No creditable response"
                }
            },
            "language": {
                "total_marks": 20,
                "assessment_criteria": ["Organisation of ideas", "Clarity of expression", "Accuracy of language"],
                "bands": {
                    "Band 5 (17-20)": "Coherent and cohesive presentation of ideas across the whole of the response; Effective use of ambitious vocabulary and grammar structures; Complex vocabulary, grammar, punctuation and spelling used accurately",
                    "Band 4 (13-16)": "Coherent presentation of ideas with some cohesion between paragraphs; Vocabulary and grammar structures sufficiently varied to convey shades of meaning; Vocabulary, grammar, punctuation and spelling used mostly accurately",
                    "Band 3 (9-12)": "Most ideas coherently presented with some cohesion within paragraphs; Vocabulary and grammar structures sufficiently varied to convey intended meaning; Vocabulary, grammar, punctuation and spelling often used accurately",
                    "Band 2 (5-8)": "Some ideas coherently presented with attempts at achieving cohesion; Mostly simple vocabulary and grammar structures used; meaning is usually clear; Vocabulary, grammar, punctuation and spelling used with varying degrees of accuracy",
                    "Band 1 (1-4)": "Ideas presented in isolation; Simple vocabulary and grammar structures used; A few examples of correct use of vocabulary, grammar, punctuation and spelling",
                    "Band 0 (0)": "No creditable response"
                }
            }
        }
    }
}
"""

        if section == "section_c" or section is None:
            base_prompt += """

FOR PAPER 1 SECTION C (CONTINUOUS WRITING):
Use the OFFICIAL GCE O-Level marking scheme: Content (10 marks) + Language (20 marks) = 30 marks

Return JSON with this EXACT structure:
{
    "section_c": {
        "title": "Continuous Writing",
        "total_marks": 30,
        "marking_rubric": {
            "content": {
                "total_marks": 10,
                "assessment_criterion": "Addressing the task",
                "bands": {
                    "Band 5 (9-10)": "All aspects of the task are fully addressed and developed in detail",
                    "Band 4 (7-8)": "All aspects of the task are addressed with some development",
                    "Band 3 (5-6)": "Some aspects of the task are addressed with some development",
                    "Band 2 (3-4)": "Some aspects of the task are addressed",
                    "Band 1 (1-2)": "Some attempts to address the task",
                    "Band 0 (0)": "No creditable response"
                }
            },
            "language": {
                "total_marks": 20,
                "assessment_criteria": ["Organisation of ideas", "Clarity of expression", "Accuracy of language"],
                "bands": {
                    "Band 5 (17-20)": "Coherent and cohesive presentation of ideas across the whole of the response; Effective use of ambitious vocabulary and grammar structures; Complex vocabulary, grammar, punctuation and spelling used accurately",
                    "Band 4 (13-16)": "Coherent presentation of ideas with some cohesion between paragraphs; Vocabulary and grammar structures sufficiently varied to convey shades of meaning; Vocabulary, grammar, punctuation and spelling used mostly accurately",
                    "Band 3 (9-12)": "Most ideas coherently presented with some cohesion within paragraphs; Vocabulary and grammar structures sufficiently varied to convey intended meaning; Vocabulary, grammar, punctuation and spelling often used accurately",
                    "Band 2 (5-8)": "Some ideas coherently presented with attempts at achieving cohesion; Mostly simple vocabulary and grammar structures used; meaning is usually clear; Vocabulary, grammar, punctuation and spelling used with varying degrees of accuracy",
                    "Band 1 (1-4)": "Ideas presented in isolation; Simple vocabulary and grammar structures used; A few examples of correct use of vocabulary, grammar, punctuation and spelling",
                    "Band 0 (0)": "No creditable response"
                }
            }
        },
        "common_pitfalls": {
            "narrative": "Over-describing settings/actions without reflection; weak or rushed endings; lack of personal voice",
            "discursive": "One-sided arguments without acknowledging counter-points; lack of concrete examples; weak conclusions",
            "descriptive": "Listing features without imagery; lack of sensory details; telling rather than showing",
            "reflective": "Surface-level reflection; lack of specific incidents; missing the 'so what' factor"
        },
        "prompts_overview": [
            {
                "number": 1,
                "prompt_text": "The full prompt text",
                "implied_genre": "narrative/descriptive/expository/argumentative/reflective",
                "possible_content_angles": [
                    "Angle 1: [Specific approach with brief description]",
                    "Angle 2: [Alternative perspective or interpretation]",
                    "Angle 3: [Another valid approach]"
                ],
                "things_to_consider": [
                    "Key question 1 the student should think about",
                    "Key question 2 for depth and reflection"
                ],
                "what_markers_look_for": [
                    "Specific element markers expect to see",
                    "Another marking focus point"
                ]
            },
            {
                "number": 2,
                "prompt_text": "The full prompt text",
                "implied_genre": "...",
                "possible_content_angles": ["..."],
                "things_to_consider": ["..."],
                "what_markers_look_for": ["..."]
            },
            {
                "number": 3,
                "prompt_text": "The full prompt text",
                "implied_genre": "...",
                "possible_content_angles": ["..."],
                "things_to_consider": ["..."],
                "what_markers_look_for": ["..."]
            },
            {
                "number": 4,
                "prompt_text": "The full prompt text",
                "implied_genre": "...",
                "possible_content_angles": ["..."],
                "things_to_consider": ["..."],
                "what_markers_look_for": ["..."]
            }
        ]
    }
}

IMPORTANT: Provide detailed content angles and considerations for ALL 4 prompts.
"""

    elif paper_format == "paper_2":
        if section == "section_a" or section is None:
            base_prompt += """

FOR PAPER 2 SECTION A (VISUAL TEXT):
CRITICAL: Q1(a) and Q1(b) answers must be EXACT PHRASES directly lifted from the visual - NOT paraphrases.

Return JSON with:
{
    "section_a": {
        "title": "Visual Text Comprehension",
        "total_marks": 5,
        "questions": [
            {
                "number": "1(a)",
                "question": "The question text",
                "answer": "EXACT phrase from visual (direct lift, not paraphrase)",
                "marks": 1,
                "source": "Exact location in visual (e.g., 'headline banner', 'program list')",
                "is_direct_retrieval": true,
                "accept": ["Only exact alternative phrases from the visual"]
            },
            {
                "number": "1(b)",
                "question": "The question text",
                "answer": "EXACT phrase from visual (direct lift, not paraphrase)",
                "marks": 1,
                "source": "Exact location in visual",
                "is_direct_retrieval": true,
                "accept": ["Only exact alternative phrases from the visual"]
            },
            {
                "number": "2",
                "question": "Question about persuasive technique",
                "answer": "Explanation of how the quoted phrase persuades...",
                "marks": 1,
                "quoted_phrase": "The exact phrase quoted in the question",
                "technique": "Name of persuasive technique (e.g., emotional appeal, inclusive language)"
            },
            {
                "number": "3",
                "question": "Question about language effect",
                "answer": "Explanation of the effect on the reader...",
                "marks": 1,
                "quoted_phrase": "The exact phrase quoted in the question"
            },
            {
                "number": "4",
                "question": "Inference question",
                "answer": "Detailed inference with SPECIFIC evidence from the visual",
                "marks": 1,
                "evidence_required": ["List of specific visual elements that support the inference"],
                "depth_notes": "Answer should reference specific programs, initiatives, or details from the visual - not just surface-level observations"
            }
        ]
    }
}

IMPORTANT FOR Q1(a) and Q1(b):
- The answer MUST be an exact quote from the visual
- Do NOT paraphrase or rephrase
- WRONG: "Events DC is committed to enhancing community life" (paraphrase)
- CORRECT: "Fall into Fun in the District" (exact phrase from visual)
"""

        if section == "section_b" or section is None:
            base_prompt += """

FOR PAPER 2 SECTION B (READING COMPREHENSION):
NOTE: Questions are numbered Q5-Q10 (continuing from Section A's Q1-Q4).

Return JSON with:
{
    "section_b": {
        "title": "Reading Comprehension",
        "total_marks": 20,
        "questions": [
            {
                "number": "5",
                "question": "The question text",
                "answer": "The model answer",
                "marks": 2,
                "question_type": "literal/inferential/vocabulary/language_effect/evaluative",
                "source_paragraph": "Paragraph X",
                "accept": ["Alternative acceptable answers"],
                "mark_breakdown": {
                    "description": "For 2-3 mark questions, provide detailed breakdown",
                    "components": [
                        {"component": "First point/element", "marks": 1},
                        {"component": "Second point/element", "marks": 1}
                    ]
                },
                "zero_marks": "Award 0 marks if: answer is completely off-topic or shows no understanding",
                "partial_marks": "Award partial marks if: identifies correct area but incomplete explanation"
            }
        ],
        "flowchart": {
            "question_number": "10",
            "marks": 4,
            "marking_per_answer": "1 mark per correct answer in correct position",
            "paragraph_answer_mapping": {
                "Paragraph 2": {
                    "answer": "A",
                    "reason": "Brief explanation of why A matches Para 2",
                    "key_phrase_from_paragraph": "Quote from paragraph that proves the match"
                },
                "Paragraph 3": {
                    "answer": "C",
                    "reason": "Brief explanation of why C matches Para 3",
                    "key_phrase_from_paragraph": "Quote from paragraph that proves the match"
                },
                "Paragraph 4": {
                    "answer": "D",
                    "reason": "Brief explanation of why D matches Para 4",
                    "key_phrase_from_paragraph": "Quote from paragraph that proves the match"
                },
                "Paragraph 5": {
                    "answer": "F",
                    "reason": "Brief explanation of why F matches Para 5",
                    "key_phrase_from_paragraph": "Quote from paragraph that proves the match",
                    "note": "If Para 5 is a conclusion/summary, note this and explain why the option still applies uniquely"
                }
            },
            "correct_answers": ["A", "C", "D", "F"],
            "distractors": {
                "B": "Explanation of why B is wrong (e.g., contradicts passage, not mentioned)",
                "E": "Explanation of why E is wrong"
            },
            "all_options": {
                "A": "Full text of option A",
                "B": "Full text of option B (distractor)",
                "C": "Full text of option C",
                "D": "Full text of option D",
                "E": "Full text of option E (distractor)",
                "F": "Full text of option F"
            },
            "common_confusions": "Note any options that might seem applicable to multiple paragraphs and explain why they only match one",
            "marking_notes": "Award 1 mark for each correct answer in the correct position. No half marks. If student puts correct letter in wrong paragraph box, award 0 marks for that answer."
        },
        "general_marking_guidance": {
            "vocabulary_questions": "Accept synonyms that fit the context. Reject if meaning is altered.",
            "language_effect_questions": "Must identify technique AND explain effect on reader for full marks.",
            "inferential_questions": "Must provide evidence from text to support inference."
        }
    }
}
"""

        if section == "section_c" or section is None:
            base_prompt += """

FOR PAPER 2 SECTION C (GUIDED COMPREHENSION + SUMMARY):
NOTE: Questions are numbered Q11-Q15 (continuing from Section B's Q5-Q10).

Return JSON with:
{
    "section_c": {
        "title": "Guided Comprehension and Summary",
        "total_marks": 25,
        "comprehension_questions": {
            "total_marks": 10,
            "questions": [
                {
                    "number": "11",
                    "question": "Question text",
                    "answer": "Model answer with clear, direct response",
                    "marks": 2,
                    "question_type": "literal_retrieval/explanation/vocabulary/cause_effect",
                    "source_paragraph": "Paragraph X",
                    "accept": ["Alternative acceptable answers"],
                    "zero_marks": "Award 0 marks if: [specific criteria]",
                    "partial_marks": "Award partial marks if: [specific criteria]"
                }
            ]
        },
        "summary": {
            "question_number": "15",
            "total_marks": 15,
            "focus": "What the summary should focus on",
            "paragraph_range": "Paragraphs X-Y",
            "starting_line": "The 10-word starting line given to students",
            "key_points": [
                {
                    "point_number": 1,
                    "original_text": "Exact text from passage for point 1",
                    "own_words": "Significantly different paraphrase using different vocabulary and structure"
                },
                {
                    "point_number": 2,
                    "original_text": "Exact text from passage for point 2",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 3,
                    "original_text": "Exact text from passage for point 3",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 4,
                    "original_text": "Exact text from passage for point 4",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 5,
                    "original_text": "Exact text from passage for point 5",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 6,
                    "original_text": "Exact text from passage for point 6",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 7,
                    "original_text": "Exact text from passage for point 7",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 8,
                    "original_text": "Exact text from passage for point 8",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 9,
                    "original_text": "Exact text from passage for point 9",
                    "own_words": "Significantly different paraphrase"
                },
                {
                    "point_number": 10,
                    "original_text": "Exact text from passage for point 10",
                    "own_words": "Significantly different paraphrase"
                }
            ],
            "model_summary": "A complete 80-word model summary demonstrating proper paraphrasing of at least 8 key points",
            "marking_rubric": {
                "content": {
                    "total_marks": 8,
                    "guidance": "1 mark per key point captured (max 8 marks). Points must be from the specified paragraphs.",
                    "bands": {
                        "7-8": "Captures 7-8 key points accurately",
                        "5-6": "Captures 5-6 key points with minor omissions",
                        "3-4": "Captures 3-4 key points",
                        "1-2": "Captures 1-2 key points",
                        "0": "No relevant content"
                    }
                },
                "style": {
                    "total_marks": 7,
                    "note": "Official GCE O-Level Summary Style Band Descriptors",
                    "bands": {
                        "7": "Sustained and successful use of own words and structures; Consistently well organised ideas which convey the meaning of the text clearly and precisely",
                        "5-6": "Frequent and usually appropriate use of own words and structures; Mostly well organised ideas which convey the meaning of the text clearly",
                        "3-4": "Some use of own words and structures; Some attempts at organising ideas to convey the meaning of the text",
                        "1-2": "Occasional attempts at use of own words and/or structures; Attempts at conveying the meaning of the text",
                        "0": "No creditable response"
                    }
                }
            },
            "paraphrasing_guidance": {
                "requirement": "Own words must be SIGNIFICANTLY different - not just synonym substitution",
                "example_poor": "Original: 'reduces stress levels' → Poor: 'decreases stress levels'",
                "example_good": "Original: 'reduces stress levels' → Good: 'helps people feel calmer'"
            }
        }
    }
}

IMPORTANT: You MUST provide sample paraphrases for ALL 10 key points, not just 4.
"""

    elif paper_format == "oral":
        base_prompt += """

FOR ORAL EXAMINATION:
Return JSON with:
{
    "reading_aloud": {
        "title": "Reading Aloud",
        "total_marks": 10,
        "challenging_words": [
            {
                "word": "The word",
                "pronunciation": "Phonetic guide",
                "common_errors": "What to watch for"
            }
        ],
        "assessment_criteria": {
            "pronunciation": "4 marks",
            "fluency": "3 marks",
            "expression": "3 marks"
        }
    },
    "sbc": {
        "title": "Stimulus-Based Conversation",
        "total_marks": 20,
        "questions": [
            {
                "number": "Q1",
                "question": "Question text",
                "expected_response": "What a good answer should include",
                "follow_up_probes": ["Probe 1", "Probe 2"],
                "assessment_focus": "What skill this tests"
            }
        ]
    },
    "conversation": {
        "title": "General Conversation",
        "total_marks": 20,
        "themes": [
            {
                "theme": "Theme name",
                "questions": [
                    {
                        "question": "Question text",
                        "expected_response": "Key points to look for",
                        "follow_ups": ["Follow-up 1"]
                    }
                ]
            }
        ]
    }
}
"""

    base_prompt += """

IMPORTANT INSTRUCTIONS:
1. Return ONLY valid JSON - no markdown, no explanations outside the JSON
2. Be thorough and include ALL questions/answers from the paper
3. Provide realistic, exam-standard model answers
4. Include marking notes where helpful
5. For open-ended questions, provide band descriptors or expected elements

Return the JSON answer key now:
"""

    return base_prompt


def generate_answer_key(
    paper_content: str,
    paper_format: str,
    section: Optional[str] = None,
    client: Optional[OpenAI] = None,
    section_a_error_key: Optional[Dict[str, Any]] = None,
) -> AnswerKey:
    """Generate an answer key for the given paper content.

    Args:
        paper_content: The full text content of the generated paper
        paper_format: paper_1, paper_2, or oral
        section: Optional specific section to generate answers for
        client: Optional OpenAI client instance
        section_a_error_key: Pre-extracted error key for Paper 1 Section A (if available)

    Returns:
        AnswerKey object with structured answers
    """
    llm_client = _ensure_openai_client(client)

    prompt = _build_answer_key_prompt(paper_content, paper_format, section, section_a_error_key)

    logger.info(
        "Generating answer key",
        paper_format=paper_format,
        section=section,
        content_length=len(paper_content),
    )

    try:
        response = llm_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert GCE O-Level English examiner creating official marking schemes. "
                        "Always return valid JSON. Be precise and thorough."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **ANSWER_KEY_LLM_PARAMS,
        )

        raw_response = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = raw_response

        # Parse JSON
        try:
            answers = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse answer key JSON: {e}")
            logger.debug(f"Raw response: {raw_response[:1000]}")
            # Return a structured error response
            answers = {
                "error": "Failed to parse answer key",
                "raw_response": raw_response[:2000],
            }

        logger.info(f"Answer key generated successfully")

        return AnswerKey(
            paper_format=paper_format,
            section=section,
            answers=answers,
        )

    except Exception as exc:
        raise AnswerKeyError(f"Failed to generate answer key: {exc}") from exc


def save_answer_key_json(answer_key: AnswerKey, output_path: Path) -> Path:
    """Save answer key to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(answer_key.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Answer key saved to {output_path}")
    return output_path


def render_answer_key_pdf(answer_key: AnswerKey, output_path: Path) -> Path:
    """Render answer key to a PDF file."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="GCE English Answer Key",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )
    normal = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    story = []

    # Title
    paper_name = answer_key.paper_format.replace("_", " ").title()
    story.append(Paragraph(f"ANSWER KEY - {paper_name}", title_style))
    if answer_key.section:
        story.append(Paragraph(f"Section: {answer_key.section.replace('_', ' ').title()}", normal))
    story.append(Paragraph(f"Generated: {answer_key.created_at.strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 10))

    # Process answers based on paper format
    answers = answer_key.answers

    def _add_section(section_key: str, section_data: dict):
        if not isinstance(section_data, dict):
            return

        title = section_data.get("title", section_key.replace("_", " ").title())
        total_marks = section_data.get("total_marks", "")

        story.append(Paragraph(f"{title} [{total_marks} marks]", section_style))

        # Handle different section types
        if "errors" in section_data:  # Paper 1 Section A
            story.append(Paragraph("<b>Editing Corrections:</b>", normal))
            for error in section_data.get("errors", []):
                line = error.get("line", "?")
                err = error.get("error", "")
                correction = error.get("correction", "")
                explanation = error.get("explanation", "")
                story.append(Paragraph(
                    f"Line {line}: <b>{err}</b> → <b>{correction}</b>",
                    normal
                ))
                if explanation:
                    story.append(Paragraph(f"   ({explanation})", normal))

            correct_lines = section_data.get("correct_lines", [])
            if correct_lines:
                story.append(Paragraph(f"<b>Correct lines:</b> {', '.join(map(str, correct_lines))}", normal))

        elif "questions" in section_data:  # Comprehension questions
            questions_list = section_data.get("questions", [])
            # Handle nested structure for Section C
            if isinstance(questions_list, dict) and "questions" in questions_list:
                questions_list = questions_list.get("questions", [])

            for q in questions_list:
                num = q.get("number", "?")
                question = q.get("question", "")[:100]
                answer = q.get("answer", "")
                marks = q.get("marks", "")
                story.append(Paragraph(f"<b>Q{num}</b> [{marks} mark(s)]", normal))
                story.append(Paragraph(f"Answer: {answer}", normal))

                # Mark breakdown for 2-3 mark questions
                mark_breakdown = q.get("mark_breakdown", {})
                if mark_breakdown and marks > 1:
                    components = mark_breakdown.get("components", [])
                    if components:
                        story.append(Paragraph("<i>Mark Breakdown:</i>", normal))
                        for comp in components:
                            comp_name = comp.get("component", "")
                            comp_marks = comp.get("marks", "")
                            story.append(Paragraph(f"   • {comp_name}: {comp_marks} mark(s)", normal))

                # Zero marks guidance
                zero_marks = q.get("zero_marks", "")
                if zero_marks:
                    story.append(Paragraph(f"<i>0 marks:</i> {zero_marks}", normal))

                # Partial marks guidance
                partial_marks = q.get("partial_marks", "")
                if partial_marks:
                    story.append(Paragraph(f"<i>Partial marks:</i> {partial_marks}", normal))

                accept = q.get("accept", [])
                if accept:
                    story.append(Paragraph(f"Also accept: {', '.join(accept)}", normal))
                story.append(Spacer(1, 4))

            # Handle flowchart if present in section
            flowchart = section_data.get("flowchart", {})
            if flowchart:
                story.append(Paragraph("<b>Flowchart Answer (Q10):</b>", normal))
                para_mapping = flowchart.get("paragraph_answer_mapping", {})
                if para_mapping:
                    for para, answer in para_mapping.items():
                        story.append(Paragraph(f"   {para}: {answer}", normal))
                else:
                    correct_seq = flowchart.get("correct_answers", flowchart.get("correct_sequence", []))
                    story.append(Paragraph(f"   Correct answers: {', '.join(correct_seq)}", normal))

                distractors = flowchart.get("distractors", [])
                if distractors:
                    story.append(Paragraph(f"   Distractors (incorrect): {', '.join(distractors)}", normal))
                story.append(Spacer(1, 4))

            # General marking guidance
            guidance = section_data.get("general_marking_guidance", {})
            if guidance:
                story.append(Paragraph("<b>General Marking Guidance:</b>", normal))
                for qtype, guide in guidance.items():
                    story.append(Paragraph(f"   <i>{qtype.replace('_', ' ').title()}:</i> {guide}", normal))

        elif "key_points" in section_data:  # Situational writing
            story.append(Paragraph("<b>Key Points to Address:</b>", normal))
            for kp in section_data.get("key_points", []):
                point = kp.get("point", "")
                story.append(Paragraph(f"• {point}", normal))
            
            # Display marking rubric for Section B (Task Fulfilment + Language)
            rubric = section_data.get("marking_rubric", {})
            if rubric:
                story.append(Spacer(1, 6))
                
                # Task Fulfilment rubric (check both spellings)
                task_rubric = rubric.get("task_fulfilment", rubric.get("task_fulfillment", {}))
                if task_rubric:
                    story.append(Paragraph(f"<b>Task Fulfilment ({task_rubric.get('total_marks', 10)} marks)</b>", normal))
                    criteria = task_rubric.get("assessment_criteria", [])
                    if criteria:
                        story.append(Paragraph("<i>Assessment Criteria:</i> " + "; ".join(criteria), normal))
                    for band, desc in task_rubric.get("bands", {}).items():
                        story.append(Paragraph(f"<b>{band}:</b> {desc}", normal))
                    story.append(Spacer(1, 4))
                
                # Language rubric
                lang_rubric = rubric.get("language", {})
                if lang_rubric:
                    story.append(Paragraph(f"<b>Language ({lang_rubric.get('total_marks', 20)} marks)</b>", normal))
                    criteria = lang_rubric.get("assessment_criteria", [])
                    if criteria:
                        story.append(Paragraph("<i>Assessment Criteria:</i> " + "; ".join(criteria), normal))
                    for band, desc in lang_rubric.get("bands", {}).items():
                        story.append(Paragraph(f"<b>{band}:</b> {desc}", normal))
                    story.append(Spacer(1, 4))

        elif "marking_rubric" in section_data:  # Continuous writing with marking rubric
            rubric = section_data.get("marking_rubric", {})

            # Content rubric (10 marks per official syllabus)
            content_rubric = rubric.get("content", {})
            if content_rubric:
                story.append(Paragraph(f"<b>Content ({content_rubric.get('total_marks', 10)} marks)</b>", normal))
                for band, desc in content_rubric.get("bands", {}).items():
                    story.append(Paragraph(f"<b>{band}:</b> {desc}", normal))
                story.append(Spacer(1, 4))

            # Language rubric (20 marks per official syllabus)
            lang_rubric = rubric.get("language", {})
            if lang_rubric:
                story.append(Paragraph(f"<b>Language ({lang_rubric.get('total_marks', 20)} marks)</b>", normal))
                for band, desc in lang_rubric.get("bands", {}).items():
                    story.append(Paragraph(f"<b>{band}:</b> {desc}", normal))
                story.append(Spacer(1, 4))

            # Common pitfalls
            pitfalls = section_data.get("common_pitfalls", {})
            if pitfalls:
                story.append(Paragraph("<b>Common Pitfalls by Question Type:</b>", normal))
                for qtype, pitfall in pitfalls.items():
                    story.append(Paragraph(f"<b>{qtype.title()}:</b> {pitfall}", normal))
                story.append(Spacer(1, 4))

            # Prompts overview with content angles and considerations
            prompts_overview = section_data.get("prompts_overview", [])
            if prompts_overview:
                story.append(Paragraph("<b>Prompts Overview:</b>", normal))
                for prompt in prompts_overview:
                    num = prompt.get("number", "?")
                    prompt_text = prompt.get("prompt_text", prompt.get("prompt_summary", ""))
                    genre = prompt.get("implied_genre", "")
                    
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(f"<b>Prompt {num}</b> ({genre})", normal))
                    if prompt_text:
                        story.append(Paragraph(f"<i>\"{prompt_text}\"</i>", normal))
                    
                    # Possible content angles
                    angles = prompt.get("possible_content_angles", [])
                    if angles:
                        story.append(Paragraph("<b>Possible Content Angles:</b>", normal))
                        for angle in angles:
                            story.append(Paragraph(f"   • {angle}", normal))
                    
                    # Things to consider
                    considerations = prompt.get("things_to_consider", [])
                    if considerations:
                        story.append(Paragraph("<b>Things to Consider:</b>", normal))
                        for consideration in considerations:
                            story.append(Paragraph(f"   • {consideration}", normal))
                    
                    # What markers look for
                    markers_look_for = prompt.get("what_markers_look_for", prompt.get("key_focus_areas", []))
                    if markers_look_for:
                        story.append(Paragraph("<b>What Markers Look For:</b>", normal))
                        for item in markers_look_for:
                            story.append(Paragraph(f"   • {item}", normal))
                    
                    story.append(Spacer(1, 4))

        elif "prompts" in section_data:  # Continuous writing (legacy format)
            story.append(Paragraph("<b>Prompt Analysis:</b>", normal))
            for prompt in section_data.get("prompts", []):
                num = prompt.get("number", "?")
                genre = prompt.get("genre", "")
                approach = prompt.get("suggested_approach", "")
                story.append(Paragraph(f"<b>Prompt {num}</b> ({genre})", normal))
                story.append(Paragraph(f"Approach: {approach}", normal))
                story.append(Spacer(1, 4))

        elif "summary" in section_data:  # Summary section
            summary = section_data.get("summary", {})

            # Comprehension questions before summary (for Section C)
            comp_questions = section_data.get("comprehension_questions", {})
            if comp_questions:
                comp_q_list = comp_questions.get("questions", []) if isinstance(comp_questions, dict) else comp_questions
                if comp_q_list:
                    story.append(Paragraph("<b>Comprehension Questions:</b>", normal))
                    for q in comp_q_list:
                        num = q.get("number", "?")
                        answer = q.get("answer", "")
                        marks = q.get("marks", "")
                        story.append(Paragraph(f"<b>Q{num}</b> [{marks} mark(s)]: {answer}", normal))

                        zero_marks = q.get("zero_marks", "")
                        if zero_marks:
                            story.append(Paragraph(f"   <i>0 marks:</i> {zero_marks}", normal))

                        partial_marks = q.get("partial_marks", "")
                        if partial_marks:
                            story.append(Paragraph(f"   <i>Partial:</i> {partial_marks}", normal))
                    story.append(Spacer(1, 6))

            story.append(Paragraph("<b>Summary Key Points:</b>", normal))
            for kp in summary.get("key_points", []):
                num = kp.get("point_number", "?")
                original = kp.get("original_text", "")
                own_words = kp.get("own_words", "")
                story.append(Paragraph(f"{num}. Original: \"{original[:100]}{'...' if len(original) > 100 else ''}\"", normal))
                story.append(Paragraph(f"   <b>Own words:</b> \"{own_words}\"", normal))

            model = summary.get("model_summary", "")
            if model:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Model Summary:</b>", normal))
                story.append(Paragraph(model, normal))

            # Summary marking rubric
            summary_rubric = summary.get("marking_rubric", {})
            if summary_rubric:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Summary Marking Rubric:</b>", normal))

                content_rubric = summary_rubric.get("content", {})
                if content_rubric:
                    story.append(Paragraph(f"<b>Content ({content_rubric.get('total_marks', 8)} marks):</b>", normal))
                    for band, desc in content_rubric.get("bands", {}).items():
                        story.append(Paragraph(f"   {band}: {desc}", normal))

                style_rubric = summary_rubric.get("style", {}) or summary_rubric.get("language", {})
                if style_rubric:
                    story.append(Paragraph(f"<b>Style ({style_rubric.get('total_marks', 7)} marks):</b>", normal))
                    for band, desc in style_rubric.get("bands", {}).items():
                        story.append(Paragraph(f"   {band}: {desc}", normal))

            # Paraphrasing guidance
            para_guidance = summary.get("paraphrasing_guidance", {})
            if para_guidance:
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Paraphrasing Guidance:</b>", normal))
                req = para_guidance.get("requirement", "")
                if req:
                    story.append(Paragraph(f"   {req}", normal))
                poor = para_guidance.get("example_poor", "")
                if poor:
                    story.append(Paragraph(f"   <i>Poor example:</i> {poor}", normal))
                good = para_guidance.get("example_good", "")
                if good:
                    story.append(Paragraph(f"   <i>Good example:</i> {good}", normal))

        elif "flowchart" in section_data:  # Flowchart question
            flowchart = section_data.get("flowchart", {})
            correct_seq = flowchart.get("correct_sequence", [])
            story.append(Paragraph(f"<b>Flowchart Answer:</b> {', '.join(correct_seq)}", normal))

        story.append(Spacer(1, 8))

    # Process each section in the answers
    for key, value in answers.items():
        if isinstance(value, dict) and key != "error":
            _add_section(key, value)

    # Handle error case
    if "error" in answers:
        story.append(Paragraph(f"<b>Error:</b> {answers['error']}", normal))

    doc.build(story)
    logger.info(f"Answer key PDF saved to {output_path}")
    return output_path
