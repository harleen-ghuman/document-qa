# Zania Document QA

A question-answering service for PDF and JSON documents. Upload a document and a list of questions to get answers with source citations. Built with FastAPI, LangChain, OpenAI (`gpt-4o-mini`), an in-memory Qdrant vector store, and a React frontend.

## Run with Docker

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Never commit API keys.

```bash
docker build -t zania-document-qa .
docker run --rm --env-file .env -p 127.0.0.1:8001:8000 zania-document-qa
```

- UI: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)
- API docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

The container runs as a non-root user and includes an API health check.

Optional: set `ENABLE_QUERY_REWRITE=true` in `.env` to add an LLM query-rewrite step. It is off by default to reduce cost and latency.

## Local development

Requires Python 3.14 and Node.js/npm.

```bash
cp .env.example .env
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Set `OPENAI_API_KEY` in `.env` before starting the backend.

Start the frontend in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. Its development proxy forwards `/api` requests to FastAPI on port 8000.

## API

### GET /api/health

Returns `{"status": "UP"}`.

### POST /api/answer

Multipart upload with two files:

- `questions_file`: JSON with a non-empty `questions` list, or a bare list of question strings.
- `document_file`: a PDF or JSON document.

```bash
curl -X POST http://127.0.0.1:8001/api/answer \
  -F "questions_file=@samples/sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your document's path. Use port 8000 for the local backend.

Example response, with answers in input order:

```json
{
  "results": [
    {
      "question": "Which cloud provider hosts the service?",
      "answer": "The service is hosted on AWS.",
      "citations": [{"source": "report.pdf", "page": 13}]
    }
  ]
}
```

- Add `?include_evidence=true` to also return `chunk_count` and full retrieved passages.
- When no supporting passage is identified, the answer is `Not found in the provided document.` with empty `citations`.
- Citation pages use PDF file order, starting at 1. JSON sources have no page number.

[Sample output](samples/sample_output.json) contains the actual response to the five challenge questions against the OneTrust SOC 2 report.

## How it works

1. Parse the document with pypdf or JSON and split it into overlapping chunks.
2. Index `text-embedding-3-small` embeddings in in-memory Qdrant.
3. Retrieve with semantic and keyword search. Multipart questions separated by question marks are also searched by component.
4. Expand PDF matches to pages and limit the context to six passages and 24,000 characters.
5. Generate a structured answer and validate supporting passage references. At most one review call is made if reference validation fails or the model abstains despite available passages.

Up to three questions run concurrently per request.

## Limits and errors

- Up to 20 questions, each at most 1,000 characters.
- Questions file: 100,000 bytes.
- Document file: 10,000,000 bytes.
- PDF: up to 200 pages.
- Extracted text: 500,000 characters.
- Index: up to 1,000 chunks.
- JSON nesting: up to 64 levels.
- Processing deadline: 120 seconds after uploads are read.

Errors return JSON with a `detail` message:

- 400/415: invalid or unsupported input.
- 413: size or processing limit exceeded.
- 502: provider response or grounding-reference failure.
- 503: provider connection, rate, or quota failure.
- 504: provider or processing timeout.
- 500: unexpected error, with a request ID.

## Observability

Single-line JSON application logs record request IDs, status, HTTP duration, chat-model token usage, and per-question retrieval and generation latency. Token counts exclude embeddings. Responses include an `X-Request-ID` header.

Per-question retrieval timing includes optional rewriting and context preparation; generation timing includes any review call. Initial parsing and indexing are included in HTTP duration.

## Tests

```bash
python -m pytest -q
```

All 52 backend tests passed, including mocked API requests, validation, parsing, limits, concurrency, citations, unexpected-error handling, and multipart retrieval.

Provider calls are mocked, so the tests do not measure real-model accuracy.

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

The frontend build and lint checks passed. Live requests were also verified through the Docker interface.

## Known limitations

- Indexes are rebuilt per request without caching or persistence.
- Authentication and rate limiting are not implemented. Concurrency limits apply per request, not globally.
- Grounding validation checks passage references, not whether every answer claim is supported.
- A grounding failure on one question fails the entire request.
- PDF citations are page-level. Scanned PDFs without extractable text require OCR, which is not implemented.
- A parsing thread may continue after the processing deadline.

Document content is sent to OpenAI for processing.