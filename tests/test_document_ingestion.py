import importlib
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from app.main import app
from app.services import ocr
from app.services import paper_generator


client = TestClient(app)
documents_router_module = importlib.import_module("app.api.documents.router")


@pytest.fixture(autouse=True)
def override_storage(tmp_path: Path):
    original_storage_root = settings.storage_root
    original_output_dir = settings.ocr_output_dir
    original_tmp_dir = settings.temp_dir
    original_paper_dir = settings.paper_output_dir

    settings.storage_root = tmp_path
    settings.ocr_output_dir = tmp_path / "texts"
    settings.temp_dir = tmp_path / "tmp"
    settings.paper_output_dir = tmp_path / "papers"
    settings.ensure_directories()

    try:
        yield
    finally:
        settings.storage_root = original_storage_root
        settings.ocr_output_dir = original_output_dir
        settings.temp_dir = original_tmp_dir
        settings.paper_output_dir = original_paper_dir


def test_ingest_document_success(monkeypatch: pytest.MonkeyPatch):
    def fake_extract_text(pdf_bytes: bytes, output_path: Path, language: str = "eng", dpi: int = 300):
        output_path.write_text("Mock OCR text", encoding="utf-8")
        return ocr.OCRResult(text="Mock OCR text", page_count=2, output_path=output_path)

    monkeypatch.setattr(documents_router_module, "extract_text_from_pdf", fake_extract_text)

    files = {"file": ("sample.pdf", b"%PDF-1.4\n...", "application/pdf")}
    response = client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"] == "sample.pdf"
    assert data["page_count"] == 2
    assert data["character_count"] == len("Mock OCR text")
    assert data["language"] == "eng"

    saved_path = settings.storage_root / data["text_path"]
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "Mock OCR text"


def test_ingest_document_rejects_non_pdf():
    files = {"file": ("notes.txt", b"Not a PDF", "text/plain")}
    response = client.post("/documents/ingest", files=files)

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF uploads are supported"


def test_ingest_document_empty_file():
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/documents/ingest", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_ingest_document_handles_ocr_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_extract_text(*args, **kwargs):
        raise ocr.OCRExtractionError("OCR failed")

    monkeypatch.setattr(documents_router_module, "extract_text_from_pdf", fake_extract_text)

    files = {"file": ("sample.pdf", b"%PDF-1.4\n...", "application/pdf")}
    response = client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    assert response.json()["detail"] == "OCR failed"


def test_generate_paper_success(monkeypatch: pytest.MonkeyPatch):
    def fake_generate_paper(**kwargs: object):
        assert kwargs.get("section") is None
        pdf_path = settings.paper_output_dir / "mock-paper.pdf"
        text_path = settings.paper_output_dir / "mock-paper.txt"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%mock")
        text_content = "Section A\nQuestion 1: Sample prompt.\nQuestion 2: Sample prompt."
        text_path.write_text(text_content, encoding="utf-8")
        return paper_generator.PaperGenerationResult(
            content=text_content,
            prompt="prompt",
            pdf_path=pdf_path,
            text_path=text_path,
            created_at=datetime(2024, 1, 1),
            section=None,
        )

    monkeypatch.setattr(documents_router_module, "generate_paper", fake_generate_paper)

    payload = {
        "difficulty": "foundational",
        "paper_format": "paper_1",
        "topics": ["Narrative writing"],
        "additional_instructions": "Include one summary question.",
    }
    response = client.post("/documents/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["difficulty"] == payload["difficulty"]
    assert data["paper_format"] == payload["paper_format"]
    assert data["section"] is None
    assert data["pdf_path"].endswith(".pdf")
    assert data["text_path"].endswith(".txt")
    assert "Section A" in data["preview"]

    pdf_file = settings.storage_root / data["pdf_path"]
    text_file = settings.storage_root / data["text_path"]
    assert pdf_file.exists()
    assert text_file.exists()
    assert text_file.read_text(encoding="utf-8").startswith("Section A")


def test_generate_paper_handles_errors(monkeypatch: pytest.MonkeyPatch):
    def failing_generate_paper(**_: object):
        raise paper_generator.PaperGenerationError("LLM offline")

    monkeypatch.setattr(documents_router_module, "generate_paper", failing_generate_paper)

    payload = {
        "difficulty": "standard",
        "paper_format": "paper_2",
    }
    response = client.post("/documents/generate", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "LLM offline"


def test_generate_section_success(monkeypatch: pytest.MonkeyPatch):
    def fake_generate_paper(**kwargs: object):
        assert kwargs.get("section") == "section_b"
        pdf_path = settings.paper_output_dir / "mock-paper-1B.pdf"
        text_path = settings.paper_output_dir / "mock-paper-1B.txt"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%mock")
        text_content = "Section B Task\nWrite a situational response."
        text_path.write_text(text_content, encoding="utf-8")
        return paper_generator.PaperGenerationResult(
            content=text_content,
            prompt="prompt",
            pdf_path=pdf_path,
            text_path=text_path,
            created_at=datetime(2024, 1, 1),
            section="section_b",
        )

    monkeypatch.setattr(documents_router_module, "generate_paper", fake_generate_paper)

    payload = {
        "difficulty": "standard",
        "paper_format": "paper_1",
        "section": "section_b",
    }
    response = client.post("/documents/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["section"] == "section_b"
    pdf_file = settings.storage_root / data["pdf_path"]
    text_file = settings.storage_root / data["text_path"]
    assert pdf_file.exists()
    assert text_file.exists()


def test_generate_section_invalid_for_oral():
    payload = {
        "difficulty": "standard",
        "paper_format": "oral",
        "section": "section_a",
    }
    response = client.post("/documents/generate", json=payload)
    assert response.status_code == 422

