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

    @model_validator(mode="after")
    def validate_section(self) -> "PaperGenerationRequest":
        if self.section and self.paper_format not in {PaperFormat.paper_1, PaperFormat.paper_2}:
            raise ValueError("Sections are only available for Paper 1 and Paper 2 formats")
        return self


class PaperGenerationResponse(BaseModel):
    difficulty: DifficultyLevel = Field(..., description="Difficulty used to generate the paper")
    paper_format: PaperFormat = Field(..., description="Format used to generate the paper")
    section: Optional["PaperSection"] = Field(
        default=None,
        description="Section that was generated (if applicable)",
    )
    pdf_path: str = Field(..., description="Relative path to the generated PDF")
    text_path: str = Field(..., description="Relative path to the generated text file")
    created_at: datetime = Field(..., description="Timestamp of paper generation (UTC)")
    preview: str = Field(..., description="First few lines of the generated paper content")
    visual_meta: Optional[dict] = Field(default=None, description="Metadata for embedded visual (url, title, host)")


class PaperSection(str, Enum):
    section_a = "section_a"
    section_b = "section_b"
    section_c = "section_c"


