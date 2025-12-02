# API Documentation

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## Table of Contents

- [Document Endpoints](#document-endpoints)
  - [POST /documents/generate](#post-documentsgenerate)
  - [POST /documents/ingest](#post-documentsingest)
- [Sync Endpoints](#sync-endpoints)
  - [POST /sync](#post-sync)
  - [GET /sync/status](#get-syncstatus)
  - [POST /sync/init-db](#post-syncinit-db)
  - [GET /sync/setup-sql](#get-syncsetup-sql)
  - [DELETE /sync/embeddings](#delete-syncembeddings)

---

## Document Endpoints

### POST /documents/generate

Generate a new GCE O-Level English examination paper using AI.

**Request Body:**

```json
{
  "difficulty": "standard",
  "paper_format": "paper_1",
  "section": "section_a",
  "topics": ["technology", "environment"],
  "additional_instructions": "Focus on formal register",
  "visual_mode": "embed",
  "search_provider": "openai"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `difficulty` | string | Yes | `foundational`, `standard`, or `advanced` |
| `paper_format` | string | Yes | `paper_1`, `paper_2`, or `oral` |
| `section` | string | No | `section_a`, `section_b`, or `section_c` (generates full paper if omitted) |
| `topics` | array | No | Focus topics for the generated content |
| `additional_instructions` | string | No | Custom instructions for the LLM |
| `visual_mode` | string | No | `embed` (default), `text_only`, or `auto` |
| `search_provider` | string | No | `openai` (default), `llm`, or `hybrid` |

**Example - Generate Paper 1 Section A (Editing):**

```bash
curl -X POST http://localhost:8000/documents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "difficulty": "standard",
    "paper_format": "paper_1",
    "section": "section_a"
  }'
```

**Example - Generate Paper 1 Section B (Situational Writing):**

```bash
curl -X POST http://localhost:8000/documents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "difficulty": "foundational",
    "paper_format": "paper_1",
    "section": "section_b",
    "topics": ["travel", "tourism"]
  }'
```

**Example - Generate Paper 2 Section C (Summary):**

```bash
curl -X POST http://localhost:8000/documents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "difficulty": "advanced",
    "paper_format": "paper_2",
    "section": "section_c"
  }'
```

**Example - Generate Full Paper 1:**

```bash
curl -X POST http://localhost:8000/documents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "difficulty": "standard",
    "paper_format": "paper_1"
  }'
```

**Response:**

```json
{
  "difficulty": "standard",
  "paper_format": "paper_1",
  "section": "section_a",
  "pdf_path": "papers/paper_1-section_a-standard-20251202-183000.pdf",
  "text_path": "papers/paper_1-section_a-standard-20251202-183000.txt",
  "created_at": "2025-12-02T18:30:00.000Z",
  "preview": "Section A [10 marks]\n\nCarefully read the text below...",
  "visual_meta": null
}
```

---

### POST /documents/ingest

Upload and OCR a PDF document to extract text.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | PDF file to process |
| `language` | string | No | OCR language code (default: `eng`) |

**Example:**

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@2016_GCE-Paper-1.pdf" \
  -F "language=eng"
```

**Response:**

```json
{
  "original_filename": "2016_GCE-Paper-1.pdf",
  "text_path": "texts/2016_GCE-Paper-1-20251202-183000.txt",
  "page_count": 12,
  "character_count": 15432,
  "language": "eng"
}
```

---

## Sync Endpoints

### POST /sync

Process all papers in `storage/original_papers/` directory:
1. Run OCR on PDFs
2. Extract metadata from filenames
3. Chunk text into segments
4. Generate embeddings
5. Store in Supabase

**Request Body:**

```json
{
  "force_reprocess": false,
  "file_filter": "*Paper-1*"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `force_reprocess` | boolean | No | Re-OCR even if text exists (default: `false`) |
| `file_filter` | string | No | Glob pattern to filter files |

**Example - Sync All Papers:**

```bash
curl -X POST http://localhost:8000/sync \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example - Sync Only Paper 1 Files:**

```bash
curl -X POST http://localhost:8000/sync \
  -H "Content-Type: application/json" \
  -d '{
    "file_filter": "*Paper1*"
  }'
```

**Example - Force Re-process Everything:**

```bash
curl -X POST http://localhost:8000/sync \
  -H "Content-Type: application/json" \
  -d '{
    "force_reprocess": true
  }'
```

**Response:**

```json
{
  "total_files": 50,
  "processed_files": 42,
  "skipped_files": 8,
  "failed_files": 0,
  "total_chunks": 350,
  "total_embeddings": 350,
  "duration_seconds": 245.5,
  "file_results": [
    {
      "filename": "2016_GCE-O-LEVEL-ENGLISH-1128-Paper-1.pdf",
      "status": "success",
      "paper_type": "paper_1",
      "year": "2016",
      "chunks_created": 12,
      "error_message": null
    },
    {
      "filename": "Sec4_English_2021_SA2_Admiralty_Ans.pdf",
      "status": "skipped",
      "paper_type": null,
      "year": null,
      "chunks_created": 0,
      "error_message": "Skipped: Answer sheet or no clear Paper1/Paper2 designation"
    }
  ]
}
```

---

### GET /sync/status

Get current sync status and embedding statistics.

**Example:**

```bash
curl http://localhost:8000/sync/status
```

**Response:**

```json
{
  "original_papers_count": 50,
  "extracted_texts_count": 42,
  "embedding_stats": {
    "total_chunks": 350,
    "total_files": 42,
    "breakdown": [
      {"paper_type": "paper_1", "section": "section_a", "count": 45},
      {"paper_type": "paper_1", "section": "section_b", "count": 52},
      {"paper_type": "paper_1", "section": "section_c", "count": 48},
      {"paper_type": "paper_1", "section": null, "count": 30},
      {"paper_type": "paper_2", "section": "section_a", "count": 40},
      {"paper_type": "paper_2", "section": "section_b", "count": 65},
      {"paper_type": "paper_2", "section": "section_c", "count": 70}
    ]
  }
}
```

---

### POST /sync/init-db

Initialize or verify the database setup. If the table doesn't exist, returns the SQL to run manually.

**Example:**

```bash
curl -X POST http://localhost:8000/sync/init-db
```

**Response (Success):**

```json
{
  "status": "success",
  "message": "Database is ready! You can now use /sync"
}
```

**Response (Setup Required):**

```json
{
  "status": "setup_required",
  "message": "Please run the setup SQL in Supabase Dashboard → SQL Editor",
  "instructions": [
    "1. Go to https://supabase.com/dashboard",
    "2. Select your project",
    "3. Click 'SQL Editor' in the left sidebar",
    "4. Paste the SQL below and click 'Run'",
    "5. Call this endpoint again to verify"
  ],
  "setup_sql": "-- SQL content here..."
}
```

---

### GET /sync/setup-sql

Get the SQL needed to set up the database.

**Example:**

```bash
curl http://localhost:8000/sync/setup-sql
```

**Response:**

```json
{
  "instructions": "Run this SQL in Supabase Dashboard → SQL Editor",
  "sql": "-- Enable pgvector extension\nCREATE EXTENSION IF NOT EXISTS vector;\n\n-- Create the embeddings table\nCREATE TABLE IF NOT EXISTS paper_embeddings (...)"
}
```

---

### DELETE /sync/embeddings

Clear embeddings from the database.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_file` | string | No | Only delete embeddings from this file |

**Example - Clear All Embeddings:**

```bash
curl -X DELETE http://localhost:8000/sync/embeddings
```

**Example - Clear Specific File:**

```bash
curl -X DELETE "http://localhost:8000/sync/embeddings?source_file=2016_GCE-Paper-1.txt"
```

**Response:**

```json
{
  "status": "success",
  "deleted_count": 12,
  "source_file": "2016_GCE-Paper-1.txt"
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 415 | Unsupported Media Type (wrong file type) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (database error) |

---

## Paper Format Reference

### Paper 1 (Writing)

| Section | Marks | Description |
|---------|-------|-------------|
| Section A | 10 | Editing - 12-line passage with 8 grammatical errors |
| Section B | 30 | Situational Writing - letter/email/report/speech |
| Section C | 30 | Continuous Writing - 4 prompts, choose 1 |

### Paper 2 (Comprehension)

| Section | Marks | Description |
|---------|-------|-------------|
| Section A | 5 | Visual Text Comprehension |
| Section B | 20 | Reading Comprehension (passage + questions) |
| Section C | 25 | Guided Comprehension + Summary |

---

## Difficulty Levels

| Level | Description |
|-------|-------------|
| `foundational` | Basic vocabulary, simpler sentence structures |
| `standard` | Grade-appropriate complexity |
| `advanced` | Challenging vocabulary, complex themes |

