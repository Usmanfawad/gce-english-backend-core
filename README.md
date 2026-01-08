# GCE English Backend

FastAPI backend for generating GCE O-Level English examination papers using AI, with RAG (Retrieval-Augmented Generation) powered by past paper embeddings.

## Features

- **Paper Generation**: Generate Paper 1 (Writing) and Paper 2 (Comprehension) with AI
- **RAG Pipeline**: Uses past papers as reference for consistent tone, style, and structure
- **OCR Ingestion**: Extract text from scanned PDF papers using RapidOCR
- **Vector Search**: Supabase pgvector for semantic similarity search
- **Visual Stimulus**: Auto-fetches relevant images for situational writing tasks

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

Create a `.env` file:

```env
# OpenAI (Required)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Supabase (Required for RAG)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
```

### 3. Setup Database

Run the server and call the init endpoint:

```bash
uv run uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/sync/setup-sql` to get the SQL, and run it in Supabase Dashboard → SQL Editor.

### 4. Sync Past Papers

Place PDF papers in `storage/original_papers/` and call:

```bash
POST http://localhost:8000/sync
```

### 5. Generate Papers

```bash
POST http://localhost:8000/documents/generate
{
  "difficulty": "standard",
  "paper_format": "paper_1",
  "section": "section_a"
}
```

## API Documentation

Interactive docs available at `http://localhost:8000/docs`

See [docs/API.md](docs/API.md) for detailed endpoint documentation with examples.

## Project Structure

```
app/
├── api/
│   ├── documents/     # Paper generation & OCR endpoints
│   └── sync/          # Embedding sync endpoints
├── config/            # Settings and logging
├── db/                # Supabase/pgvector operations
├── services/
│   ├── paper_generator.py  # LLM paper generation
│   ├── embeddings.py       # Chunking & embedding
│   ├── rag.py              # RAG retrieval
│   ├── sync.py             # Sync orchestration
│   ├── ocr.py              # PDF text extraction
│   └── visuals.py          # Visual stimulus fetching
└── templates/         # HTML templates for PDF rendering

storage/
├── original_papers/   # Place source PDFs here
├── texts/             # OCR-extracted text files
└── papers/            # Generated papers (PDF, HTML, TXT)
```

## Supported File Naming

The sync system recognizes these filename patterns:

| Pattern | Example |
|---------|---------|
| GCE Official | `2016_GCE-O-LEVEL-ENGLISH-1128-Paper-1.pdf` |
| School Papers | `Sec4_English_2021_SA2_admiralty_Paper1.pdf` |

Files containing `_Ans` (answer sheets) or without `Paper1`/`Paper2` designation are automatically skipped.

## Testing

```bash
uv run pytest
```

## License

Proprietary - All rights reserved.
