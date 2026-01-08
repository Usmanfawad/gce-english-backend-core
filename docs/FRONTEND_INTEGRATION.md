## GCE English Backend – Frontend Integration Guide

This document explains how a frontend app should integrate with the GCE English backend:

- **Auth & session flow** (signup, login, logout, token storage)
- **Protected API calls** for document generation and sync
- **Error handling patterns**

Base URL examples assume `http://localhost:8000`. Adjust for your deployed environment.

---

## 1. Authentication Overview

- **Auth model**: Custom `app_users` table in Supabase with **email + password (bcrypt)**.
- **Token type**: **JWT access token**, signed with HS256.
- **Auth endpoints**: All under the `/auth` prefix.
- **Protected endpoints**: All `/documents/*` and `/sync/*` routes require a valid bearer token.

### 1.1 User signup

- **Endpoint**: `POST /auth/signup`
- **Auth required**: No
- **Purpose**: Create a new user account.
- **Password rules**:
  - Minimum: **8** characters
  - Maximum: **72** characters (bcrypt limitation)

**Request body:**

```json
{
  "email": "user@example.com",
  "password": "strongpassword",
  "full_name": "Jane Doe"
}
```

**Response body (201 Created):**

```json
{
  "id": "c6c3c2a1-1234-5678-9abc-def012345678",
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "created_at": null
}
```

**Error cases (examples):**

- `400 Bad Request` – email already exists:

  ```json
  { "detail": "A user with this email already exists" }
  ```

- `422 Unprocessable Entity` – invalid email or password too short/long.

### 1.2 User login

- **Endpoint**: `POST /auth/login`
- **Auth required**: No
- **Purpose**: Exchange email/password for a JWT access token.

**Request body:**

```json
{
  "email": "user@example.com",
  "password": "strongpassword"
}
```

**Response body (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

> **Frontend responsibility:** Store `access_token` securely (e.g. in memory or `localStorage` depending on your threat model) and include it in the `Authorization` header for all protected requests:
>
> `Authorization: Bearer <access_token>`

**Error cases (examples):**

- `400 Bad Request` – bad credentials:

  ```json
  { "detail": "Incorrect email or password" }
  ```

- `503 Service Unavailable` – Supabase connectivity or config issues:

  ```json
  { "detail": "Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_KEY." }
  ```

### 1.3 Get current user (`/auth/me`)

- **Endpoint**: `GET /auth/me`
- **Auth required**: Yes (JWT in header)
- **Purpose**: Fetch the currently authenticated user.

**Request headers:**

```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Response body (200 OK):**

```json
{
  "id": "c6c3c2a1-1234-5678-9abc-def012345678",
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "created_at": null
}
```

**Error cases (examples):**

- `401 Unauthorized` – missing/invalid/expired token:

  ```json
  { "detail": "Could not validate credentials" }
  ```

  or

  ```json
  { "detail": "Token has expired" }
  ```

---

## 2. Global Request Rules

- **Content type**:
  - JSON endpoints: `Content-Type: application/json`
  - File uploads: `multipart/form-data`
- **Auth header** for protected routes:

  ```http
  Authorization: Bearer <access_token>
  ```

- **Error format (common)**:

  ```json
  { "detail": "Error message describing what went wrong" }
  ```

Frontend should:

- Check HTTP status codes.
- Show `detail` message for user-friendly error reporting (where appropriate).
- Handle `401` by:
  - Clearing stored token.
  - Redirecting to login.

---

## 3. Health & Basic Info

### 3.1 Root endpoint

- **Endpoint**: `GET /`
- **Auth required**: No
- **Purpose**: Basic API health and metadata (can be used for environment checks).

**Response body (200 OK):**

```json
{
  "message": "Welcome to the GCE English backend",
  "version": "0.1.0",
  "status": "operational",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

The frontend can call this once on startup to:

- Confirm the backend is reachable.
- Optionally display the backend version in a debug page.

---

## 4. Document APIs (Protected)

All `/documents/*` endpoints require a valid JWT access token.

### 4.1 Generate paper

- **Endpoint**: `POST /documents/generate`
- **Auth required**: Yes
- **Purpose**: Generate a new exam paper (Paper 1 or 2, optionally a specific section).

**Request headers:**

```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request body (example – Paper 1, Section A):**

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

Key fields:

- `difficulty`: `"foundational" | "standard" | "advanced"`
- `paper_format`: `"paper_1" | "paper_2" | "oral"`
- `section` (optional): `"section_a" | "section_b" | "section_c"` (only for `paper_1` / `paper_2`)
- `visual_mode` (optional): `"embed" | "text_only" | "auto"`
- `search_provider` (optional): `"openai" | "llm" | "hybrid"`

**Response body (example):**

```json
{
  "difficulty": "standard",
  "paper_format": "paper_1",
  "section": "section_a",
  "pdf_path": "papers/paper_1-section_a-standard-20251202-183000.pdf",
  "text_path": "papers/paper_1-section_a-standard-20251202-183000.txt",
  "created_at": "2025-12-02T18:30:00.000Z",
  "preview": "Section A [10 marks]\n\nCarefully read the text below...",
  "visual_meta": null,
  "download_url": "https://<project-ref>.supabase.co/storage/v1/object/sign/Genrated_Papers/<user_id>/paper1/paper_1-section_a-standard-20251202-183000.pdf?token=..."
}
```

**PaperGenerationResponse** model:
- `pdf_path`: relative path under backend storage (legacy/local use)
- `text_path`: relative path for text version
- `download_url`: **signed** HTTP(S) URL to download the generated PDF from Supabase Storage (private bucket, time-limited)

Frontend should usually use **`download_url`** directly for file download buttons (open in new tab or set as href). No extra headers are needed for this URL.

### 4.2 Upload & OCR a PDF

- **Endpoint**: `POST /documents/ingest`
- **Auth required**: Yes
- **Purpose**: Upload a PDF, run OCR, and get back the text path and stats.
- **Content type**: `multipart/form-data`

**Form fields:**

- `file` – the PDF file to upload.
- `language` (optional) – OCR language code, default `"eng"`.

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@2016_GCE-Paper-1.pdf" \
  -F "language=eng"
```

**Response body (example):**

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

## 5. Sync APIs (Protected, Admin-Oriented)

These endpoints are primarily for administrative or backend sync tasks (not typical end-user UI), but the frontend may need to trigger them in an admin panel.

All `/sync/*` endpoints require a valid JWT access token.

### 5.1 Start sync

- **Endpoint**: `POST /sync`
- **Auth required**: Yes
- **Purpose**: Process all papers in `storage/original_papers`:
  - Run OCR
  - Extract metadata
  - Chunk text
  - Generate embeddings
  - Store in Supabase

**Request body (example):**

```json
{
  "force_reprocess": false,
  "file_filter": "*Paper-1*"
}
```

**Response body (summary):**

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
    }
  ]
}
```

### 5.2 Sync status

- **Endpoint**: `GET /sync/status`
- **Auth required**: Yes
- **Purpose**: Show current embedding stats and available OCR text files.

**Response body (example):**

```json
{
  "original_papers_count": 50,
  "extracted_texts_count": 42,
  "embedding_stats": {
    "total_chunks": 350,
    "total_files": 42,
    "breakdown": [
      { "paper_type": "paper_1", "section": "section_a", "count": 45 },
      { "paper_type": "paper_2", "section": "section_c", "count": 70 }
    ]
  }
}
```

---

## 6. Typical Frontend Flows

### 6.1 Login flow

1. **User opens login page.**
2. On submit:
   - Call `POST /auth/login` with `email` and `password`.
   - On success, store `access_token`.
   - Optionally call `GET /auth/me` to populate current user state.
3. Redirect to the main app (dashboard / generator page).

### 6.2 Signup flow

1. **User opens signup page.**
2. On submit:
   - Call `POST /auth/signup`.
   - On success, either:
     - Auto-login by calling `POST /auth/login`, or
     - Redirect to login screen with a success message.

### 6.3 Generate paper flow

1. User must be **logged in** (have a token).
2. Frontend collects generation options:
   - Difficulty
   - Paper format
   - Optional section
   - Optional topics / instructions
3. Call `POST /documents/generate` with auth header.
4. Display:
   - Preview text (`preview`)
   - Links/buttons to download the generated PDF.

### 6.4 Admin sync dashboard (optional)

1. Show sync summary using `GET /sync/status`.
2. Allow triggering sync using `POST /sync` with options.
3. Display progress and results (e.g. file counts and errors).

---

## 7. Environment & Configuration (Frontend)

Frontend should be configurable via environment variables or configuration files:

- **Backend base URL**: e.g. `VITE_API_BASE_URL=http://localhost:8000`
- **Timeouts & error handling**:
  - Global HTTP interceptor for:
    - Injecting `Authorization` header when token exists.
    - Redirecting to login on `401`.
    - Showing toast/notification on `500`/`503`.

### 7.1 Example HTTP client (pseudo-code)

```ts
// Example using fetch; adjust for your stack (Axios, React Query, etc.)

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function getAuthToken(): string | null {
  return localStorage.getItem("access_token");
}

async function apiFetch(path: string, options: RequestInit = {}) {
  const token = getAuthToken();

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {})
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    // Handle 401: logout + redirect to login
    if (res.status === 401) {
      localStorage.removeItem("access_token");
      // redirect to login page
    }
    // Surface backend "detail" if available
    throw new Error(data?.detail ?? "Request failed");
  }

  return data;
}
```

---

## 8. Summary for Frontend Team

- **Auth first**: Implement signup/login, store the **JWT access token**, and attach it to `Authorization` headers.
- **Protect everything else**: All `/documents/*` and `/sync/*` calls must include the bearer token.
- **Rely on `detail`**: Use the consistent `{ "detail": "..." }` error shape to show user-facing errors.
- **Follow sample payloads**: For paper generation and OCR, match the request/response JSON detailed above. 


