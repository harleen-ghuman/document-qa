# Zania Document QA

A document question-answering application with a FastAPI backend and React/TypeScript frontend. Supports JSON and PDF documents using LangChain, OpenAI, and an in-memory Qdrant vector database.

## Requirements

- Docker to run the complete application.
- An OpenAI API key.
- For local development: Python 3.14 and Node.js/npm.

## Configuration

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Do not commit API keys.

Query rewriting is disabled by default to reduce latency and API cost. Set `ENABLE_QUERY_REWRITE=true` to enable it, then restart the backend.

## Run with Docker

```bash
docker build -t zania-document-qa .
docker run --rm --name zania-qa --env-file .env -p 127.0.0.1:8001:8000 zania-document-qa
```

- React UI: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)
- Interactive API: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

The container serves both the frontend and backend, runs as a non-root user, and includes an API health check. No separate Qdrant server is required.

## Local development

Start the backend from the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. Its development proxy forwards `/api` requests to FastAPI on port 8000.

## Usage

Upload a PDF or JSON document, then type a question or upload a questions JSON file. Answers include expandable retrieved sources.

## API

### GET /api/health

```json
{"status": "UP"}
```

### POST /api/answer

Accepts two multipart files:

- `questions_file`: a JSON object with a `questions` list, or a bare list of question strings.
- `document_file`: a JSON or PDF source document.

Example questions file:

```json
{
  "questions": [
    "Which cloud provider hosts the service?",
    "What is the incident notification policy?"
  ]
}
```

Example request:

```bash
curl -X POST http://127.0.0.1:8001/api/answer \
  -F "questions_file=@samples/sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your document's path. Use port 8000 for the local backend.

The default response returns answers in input order with source citations:

```json
{
  "results": [
    {
      "question": "Which cloud provider hosts the service?",
      "answer": "The service is hosted on AWS.",
      "citations": [
        {
          "source": "report.pdf",
          "page": 13
        }
      ]
    }
  ]
}
```

Add `include_evidence=true` to return chunk count and full retrieved passages:

```bash
curl -X POST "http://127.0.0.1:8001/api/answer?include_evidence=true" \
  -F "questions_file=@samples/sample_questions.json" \
  -F "document_file=@report.pdf"
```

Null fields are omitted. Unsupported questions return `Not found in the provided document.` with an empty citations list.

## How it works

1. Validate uploads and parse the document in a worker thread.
2. Split content into overlapping chunks and index embeddings in Qdrant.
3. Retrieve evidence using semantic and keyword search, with optional query rewriting.
4. Expand PDF matches to pages and apply context limits.
5. Generate structured answers and validate supporting passage references.
6. Perform at most one review call for invalid references or an unsupported answer.

Uses `gpt-4o-mini` for answers and `text-embedding-3-small` for embeddings. Up to three questions run concurrently per request.

## Project structure

- `frontend/`: React/TypeScript interface.
- `main.py`: API routes, validation, frontend serving, and request logging.
- `qa_service.py`: indexing, retrieval, generation, and concurrency.
- `document_parser.py`: JSON/PDF parsing.
- `json_validation.py`: JSON nesting checks.
- `keyword_search.py`: keyword ranking.
- `grounding.py`: passage-reference validation and review.
- `schemas.py`: response and generation models.
- `request_context.py`: request ID propagation.
- `samples/`: example documents and questions.
- `tests/`: unit and mocked API tests.
- `Dockerfile`: container build and health check.

## Validation

```bash
python -m pytest -q
```

All 50 backend tests passed, covering validation, parsing, limits, retrieval, response fields, concurrency, and grounding checks. Provider calls are mocked.

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

Frontend build and lint passed. Live document queries were also verified through the Docker interface.

## Limits and error handling

- Up to 20 questions, each at most 1,000 characters.
- Questions file: 100,000 bytes.
- Document file: 10,000,000 bytes.
- PDF: up to 200 pages.
- Extracted text: 500,000 characters.
- Index: up to 1,000 chunks.
- JSON nesting: up to 64 levels.
- Context: up to six passages and 24,000 characters total.
- Processing deadline: 120 seconds after uploads are read.

Input and provider failures return appropriate JSON errors. Unexpected failures return a generic 500 response with a request ID.

Structured logs include request IDs, HTTP duration, chat-model token usage, and per-question retrieval and generation timing. Token counts exclude embeddings. Responses include `X-Request-ID`.

## Design notes

Indexes are held in memory and rebuilt per request. Passage-reference validation checks source references; answer accuracy still depends on retrieval and the model. Document content is sent to OpenAI for processing.