# DC-TIM RAG backend

This is the Python backend foundation for the existing DC-TIM frontend. It is intentionally a
modular monolith: one deployable FastAPI application with feature-owned modules and replaceable
ports for embeddings, retrieval storage, and answer generation.

PostgreSQL with pgvector is the planned persistent vector store. The Compose file uses the official
pgvector image pinned to `0.8.6-pg17-bookworm` and initializes the `vector` extension.

## Providers (Phase 1)

| Port | Dev / tests | Default hosted path |
| --- | --- | --- |
| `EmbeddingProvider` | `HashEmbeddingProvider` | OpenRouter `nvidia/nemotron-3-embed-1b:free` (dim **2048**) |
| `AnswerGenerator` | `DemoGroundedAnswerGenerator` | OpenRouter free chat (`qwen/qwen3.8-27b:free` + fallbacks on 429) |
| `VectorStore` | `InMemoryVectorStore` | `PgVectorStore` when `DATABASE_URL` is set |

Set `OPENROUTER_API_KEY` in `backend/.env` to enable free OpenRouter adapters automatically.
Without a key, the service stays on hash embeddings and the demo answer generator.

Free OpenRouter endpoints are rate-limited (about 50 requests/day without credits). After
switching embedding models or dimensions, **re-ingest** all sources — old vectors are not
compatible. Run `alembic upgrade head` so migration `0003` widens `rag_chunks.embedding` to
`vector(2048)`.

Optional paid OpenAI remains available via `RAG_EMBEDDING_PROVIDER=openai` /
`RAG_ANSWER_PROVIDER=openai` and `OPENAI_API_KEY`.

## Document storage

Uploaded files go through the `BlobStore` port. Two backends ship in
`app/modules/rag/infrastructure/storage.py`:

| Backend | Use when | `file_path` holds |
| --- | --- | --- |
| `LocalFileStorage` | local dev, or a single node with a mounted volume | `uploads/<workspace_id>/<document_id>_<filename>` |
| `CloudinaryStorage` | **any serverless deployment** | the `https://res.cloudinary.com/...` delivery URL |

Serverless filesystems are ephemeral, so on Vercel/Lambda/Cloud Run the local
backend silently loses every upload on the next cold start. Set the three
credentials and Cloudinary is selected automatically:

```dotenv
RAG_STORAGE_PROVIDER=          # blank = cloudinary when the keys below are complete
CLOUDINARY_CLOUD_NAME=i5pxe4ko
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
CLOUDINARY_FOLDER=dc-tim/uploads
CLOUDINARY_TIMEOUT_SECONDS=60  # keep below the function timeout
```

Notes:

- Uploads are **signed server-side**, so `CLOUDINARY_API_SECRET` never reaches
  the browser. Do not switch to unsigned browser uploads unless you also want
  the secret on the client.
- Documents are stored as `raw` resources, so PDFs/DOCX/XLSX stay byte-identical
  instead of being treated as images.
- Uploads are also converted into a canonical, provenance-marked text stream for
  retrieval. PDF pages and DOCX paragraphs/tables carry source markers, while the
  canonical extraction is retained in `rag_documents.extracted_text` for auditing
  and re-indexing.
- PDF extraction currently reads text layers only. Pages without a text layer are
  marked as requiring OCR and a fully scanned PDF is rejected rather than silently
  indexed as an empty document.
- DOCX extraction preserves heading/list structure, tables, headers, and footers.
- Assets are keyed `<CLOUDINARY_FOLDER>/<workspace_id>/<document_id>_<filename>`,
  which keeps tenants separated and makes re-uploads idempotent.
- With `RAG_ENVIRONMENT=production`, a `cloudinary` provider with missing
  credentials **fails at startup** rather than degrading to local disk.
- Keep `RAG_MAX_UPLOAD_MB` at or below your platform's request body limit —
  Cloudinary's own free tier caps uploads at 10 MB per file.

## Ops & security (Phase 5)

- **Rate limits**: `RAG_RATE_LIMIT_PER_MINUTE` (default 120) on `/api/v1/*`
- **Upload guards**: extension + magic-byte checks, size cap (`RAG_MAX_UPLOAD_MB`),
  extracted-text length cap (`RAG_MAX_EXTRACTED_CHARS`)
- **Untrusted evidence**: retrieved chunks are sanitized and wrapped; the LLM is told
  not to follow instructions inside documents
- **Strict grounding**: generated answers must cite valid retrieved evidence blocks;
  uncited answers are rejected, and provider failures are not replaced with generic
  policy advice
- **Query telemetry**: each `/rag/query` response includes `telemetry` with latency
  split, provider/model names, and citation coverage

### Postgres backup (local)

```powershell
docker compose exec postgres pg_dump -U dc_tim dc_tim > backup.sql
# restore:
# Get-Content backup.sql | docker compose exec -T postgres psql -U dc_tim dc_tim
```

Test restore on a scratch database before loading production policy corpora.

## Evaluation (Phase 6 gates)

A gold set of policy questions lives in `eval/gold_set.json`. It measures:

- retrieval recall@k
- answer groundedness and citation accuracy
- insufficient-evidence refusal
- latency split (embed / search / generate)
- estimated cost per answer (token heuristic; $0 for hash/demo)
- cross-workspace isolation

Deterministic hash embedding + demo generator — no API key required.

```powershell
cd backend
python -m eval
python -m pytest tests/test_eval.py -q
```

Raise the bar with hosted embeddings only after this baseline passes. Do not add hybrid
retrieval or rerankers until the gold set shows a concrete recall gap.

`RAG_MIN_SCORE` (default `0.15`) drops weak cosine hits before answer generation.
`RAG_EMBEDDING_BATCH_SIZE` (default `256`) splits large documents into provider-safe
embedding requests. Keep it at or below the provider's maximum batch size.

## Auth (Phase 3)

RAG routes require `Authorization: Bearer <token>`. Workspace isolation is derived from the JWT
`workspace_id` claim — clients cannot select another workspace via `X-Workspace-Id`.

Demo users (bcrypt-hashed at process start):

| Email | Password | Workspace |
| --- | --- | --- |
| `admin@dc-tim.ai` | `admin123` | `admin` |
| `analyst@dc-tim.ai` | `analyst123` | `analyst` |

Set `AUTH_JWT_SECRET` in production. Login: `POST /api/v1/auth/login`. Session check:
`GET /api/v1/auth/me`.

## Run locally

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
# Edit .env: set OPENROUTER_API_KEY (and DATABASE_URL if using Postgres)
docker compose up -d postgres
alembic upgrade head
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. OpenAPI is at `http://localhost:8000/docs`.
PostgreSQL is exposed on host port `5436` by default to avoid colliding with other local database
containers; its internal PostgreSQL port remains `5432`. Change `POSTGRES_PORT` in `.env` if needed.

Check the database and extension:

```powershell
docker compose ps
docker compose exec postgres psql -U dc_tim -d dc_tim -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

Stop the container while keeping data:

```powershell
docker compose down
```

The named volume is intentionally retained by `docker compose down`. Remove it only when you
explicitly want to delete local database data: `docker compose down -v`.

## API examples

Login:

```powershell
$login = Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/auth/login `
  -ContentType "application/json" `
  -Body '{"email":"admin@dc-tim.ai","password":"admin123"}'
$token = $login.access_token
```

Index a source:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/rag/ingest `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body '{"title":"Education policy","source_type":"knowledge","content":"Teacher training and school access improve learning outcomes.","metadata":{"country":"Rwanda"}}'
```

Query indexed sources:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/rag/query `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body '{"query":"What improves learning outcomes?","top_k":3}'
```

## Structure

```text
backend/
├── app/
│   ├── core/                 # configuration and request identity
│   └── modules/rag/          # one feature: API, use cases, domain, ports, adapters
└── tests/                    # fast unit tests for the RAG feature
```

See [the RAG architecture note](../docs/architecture/rag.md) and
[ADR-001](../docs/decisions/ADR-001-modular-monolith-for-rag-backend.md) for the decisions and
the next implementation stages.
