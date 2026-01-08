"""Schemas for document ingestion and generation endpoints."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, model_validator


class DocumentIngestResponse(BaseModel):
    original_filename: str = Field(..., description="Name of the uploaded PDF")
    text_path: str = Field(..., description="Relative path to the extracted text file")
    page_count: int = Field(..., ge=1, description="Number of pages processed")
    character_count: int = Field(..., ge=0, description="Number of characters extracted")
    language: str = Field(..., description="Language code used for OCR")


class DifficultyLevel(str, Enum):
    foundational = "foundational"
    standard = "standard"
    advanced = "advanced"


class PaperFormat(str, Enum):
    paper_1 = "paper_1"
    paper_2 = "paper_2"
    oral = "oral"


class PaperGenerationRequest(BaseModel):
    difficulty: DifficultyLevel = Field(..., description="Target difficulty for the generated paper")
    paper_format: PaperFormat = Field(..., description="Exam paper format to generate")
    section: Optional["PaperSection"] = Field(
        default=None,
        description="Optional section (A, B, or C) to generate individually for the selected paper",
    )
    visual_mode: Optional[Literal["embed", "text_only", "auto"]] = Field(
        default="embed",
        description="How visuals should appear in papers (embed screenshot in PDF, text description only, or auto)",
    )
    search_provider: Optional[Literal["llm", "openai", "hybrid"]] = Field(
        default="openai",
        description="How to source visuals at runtime (LLM URLs first, OpenAI web search, or hybrid)",
    )
    topics: Optional[List[str]] = Field(
        default=None,
        description="Optional list of focus topics to emphasize in the generated paper",
    )
    additional_instructions: Optional[str] = Field(
        default=None,
        description="Optional free-form instructions for the LLM",
    )

    generate_answer_key: bool = Field(
        default=False,
        description="Whether to also generate an answer key alongside the paper"
    )

    @model_validator(mode="after")
    def validate_section(self) -> "PaperGenerationRequest":
        if self.section:
            # Paper 1 and Paper 2 sections
            paper_sections = {PaperSection.section_a, PaperSection.section_b, PaperSection.section_c}
            # Oral exam sections
            oral_sections = {PaperSection.reading_aloud, PaperSection.sbc, PaperSection.conversation}

            if self.paper_format in {PaperFormat.paper_1, PaperFormat.paper_2}:
                if self.section not in paper_sections:
                    raise ValueError(f"Paper 1/2 only supports sections: {[s.value for s in paper_sections]}")
            elif self.paper_format == PaperFormat.oral:
                if self.section not in oral_sections:
                    raise ValueError(f"Oral exam only supports sections: {[s.value for s in oral_sections]}")
        return self


class PaperGenerationResponse(BaseModel):
    difficulty: DifficultyLevel = Field(..., description="Difficulty used to generate the paper")
    paper_format: PaperFormat = Field(..., description="Format used to generate the paper")
    section: Optional["PaperSection"] = Field(
        default=None,
        description="Section that was generated (if applicable)",
    )
    pdf_path: str = Field(..., description="Relative path to the generated PDF (local storage)")
    text_path: str = Field(..., description="Relative path to the generated text file")
    created_at: datetime = Field(..., description="Timestamp of paper generation (UTC)")
    preview: str = Field(..., description="First few lines of the generated paper content")
    visual_meta: Optional[dict] = Field(default=None, description="Metadata for embedded visual (url, title, host)")
    download_url: Optional[str] = Field(
        default=None,
        description="Public URL to download the generated PDF from Supabase Storage (if available)",
    )
    answer_key: Optional[dict] = Field(
        default=None,
        description="Generated answer key if requested"
    )
    answer_key_pdf_path: Optional[str] = Field(
        default=None,
        description="Path to the answer key PDF if generated"
    )


class PaperSection(str, Enum):
    section_a = "section_a"
    section_b = "section_b"
    section_c = "section_c"
    # Oral exam sections
    reading_aloud = "reading_aloud"
    sbc = "sbc"  # Stimulus-Based Conversation
    conversation = "conversation"


class AnswerKeyRequest(BaseModel):
    """Request to generate answer keys for a paper."""
    paper_content: str = Field(..., description="The generated paper content to create answers for")
    paper_format: PaperFormat = Field(..., description="Format of the paper")
    section: Optional[PaperSection] = Field(default=None, description="Section to generate answers for")
    output_format: Literal["json", "pdf", "both"] = Field(
        default="json",
        description="Output format for the answer key"
    )


class AnswerKeyResponse(BaseModel):
    """Response containing generated answer keys."""
    paper_format: PaperFormat
    section: Optional[PaperSection] = None
    answers: dict = Field(..., description="Answer key data structure")
    json_path: Optional[str] = Field(default=None, description="Path to JSON answer key file")
    pdf_path: Optional[str] = Field(default=None, description="Path to PDF answer key file")
    created_at: datetime


