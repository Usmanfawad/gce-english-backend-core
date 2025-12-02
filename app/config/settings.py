from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GCE English Backend"
    app_version: str = "0.1.0"
    app_description: str = "APIs for processing GCE English examination papers"
    cors_allow_origins: List[str] = ["*"]

    storage_root: Path = Path("storage")
    ocr_output_dir: Path = storage_root / "texts"
    temp_dir: Path = storage_root / "tmp"
    paper_output_dir: Path = storage_root / "papers"
    original_papers_dir: Path = storage_root / "original_papers"
    html_template_dir: Path = Path("app") / "templates"
    visual_output_dir: Path = storage_root / "visuals"

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    tavily_api_key: Optional[str] = None

    # Supabase configuration (REST API - no direct DB connection needed)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    # Embedding configuration
    embedding_chunk_size: int = 1000
    embedding_chunk_overlap: int = 200

    def ensure_directories(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.ocr_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.paper_output_dir.mkdir(parents=True, exist_ok=True)
        self.original_papers_dir.mkdir(parents=True, exist_ok=True)
        self.html_template_dir.mkdir(parents=True, exist_ok=True)
        self.visual_output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()