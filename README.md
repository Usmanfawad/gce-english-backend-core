# GCE English Backend

FastAPI backend for ingesting scanned GCE English examination papers and running OCR to produce searchable text files.

## Features

- `/documents/ingest` endpoint accepts PDF uploads, performs OCR (via RapidOCR/ONNXRuntime), and stores extracted text.
- Structured logging with Loguru and rotating file handlers.
- Configurable storage directories for OCR outputs and temporary files.

## Getting Started

1. Create and activate a Python 3.12 environment.
2. Install dependencies:

   ```bash
   uv pip install -e .
   # or
   pip install -e .
   ```

3. No external OCR binaries required. OCR runs locally using RapidOCR (ONNXRuntime). PDF rendering uses PDFium via `pypdfium2` and requires no external system binaries.
4. (Optional) Create a `.env` file to override defaults (see `app/config/settings.py` for available options).

## Running Locally

```bash
uv run uvicorn app.main:app --reload
```

The interactive API docs are available at `http://localhost:8000/docs`.

## Testing

```bash
uv run pytest
```

## Project Structure

- `app/api/documents/` — FastAPI routers and schemas for document ingestion.
- `app/services/ocr.py` — OCR extraction helpers using RapidOCR (ONNXRuntime).
- `app/config/` — application configuration and logging setup.
- `storage/` — default directory for generated text files (created automatically).




