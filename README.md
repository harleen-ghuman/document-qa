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
Query rewriting is disabled by default to reduce latency and API cost. Set ENABLE_QUERY_REWRITE=true in .env to enable an additional retrieval query.

## Run with Docker

```bash
docker build -t zania-document-qa .
docker run --rm --name zania-qa --env-file .env -p 127.0.0.1:8001:8000 zania-document-qa
```

- React UI: http://127.0.0.1:8001/
- Interactive API: http://127.0.0.1:8001/docs

Docker builds and serves the frontend alongside FastAPI. No separate Qdrant server is required.

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

Upload a PDF or JSON source document, then type one question or upload a questions JSON file. Answers appear with expandable retrieved sources. Expanding sources does not make another AI request.

## API

### GET /api/health

```json
{"status": "UP"}
```

### POST /api/answer

Accepts two multipart files:

- `questions_file`: a JSON object containing a non-empty list of question strings.
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

Example JSON source document:

```json
{
  "cloud_provider": "AWS",
  "incident_notification_policy": "Notify customers without undue delay."
}
```

Example request:

```bash
curl -X POST http://127.0.0.1:8001/api/answer \
  -F "questions_file=@samples/sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your source document's path. Use port 8000 for the local backend.

The default response contains ordered question-answer pairs:

```json
{
  "results": [
    {
      "question": "Which cloud provider hosts the service?",
      "answer": "The service is hosted on AWS.",
      "citations": [{"source": "document.json"}]
    }
  ]
}
```

Set `include_evidence=true` to also return `chunk_count` and `retrieved_evidence`, including source filenames, passage text, and PDF page numbers where available:

```bash
curl -X POST "http://127.0.0.1:8001/api/answer?include_evidence=true" \
  -F "questions_file=@samples/sample_questions.json" \
  -F "document_file=@report.pdf"
```

Null fields are omitted.

## How it works

1. Validate uploads and JSON nesting, then parse the document in a worker thread.
2. Split content into overlapping chunks and index embeddings in Qdrant.
3. Combine semantic retrieval, question rewriting, and BM25-style keyword search.
4. Expand PDF matches to pages and apply evidence-size limits.
5. Generate structured answers with `gpt-4o-mini` and validate supporting passage references.
6. Perform at most one additional review when references are invalid or the model abstains despite available passages.

Embeddings use `text-embedding-3-small`. Up to three questions run concurrently per request, with results returned in input order.

Passage-reference validation checks that referenced passages exist. It does not independently verify that every answer claim is supported.

## Project structure

- `frontend/`: React/TypeScript interface.
- `main.py`: routes, validation, processing deadline, frontend serving, and logging.
- `qa_service.py`: indexing, retrieval, generation, and concurrency.
- `document_parser.py`: JSON/PDF parsing and extraction limits.
- `json_validation.py`: JSON nesting checks.
- `keyword_search.py`: keyword ranking.
- `grounding.py`: passage-reference validation and bounded review.
- `schemas.py`: API and generation models.
- `tests/`: unit and mocked API tests.
- `Dockerfile`: application build and packaging.

## Tests

```bash
python -m pytest -q
```

All 50 backend test cases passed. Coverage includes parsing, invalid inputs, processing limits, deadlines, retrieval, response fields, concurrency, and passage-reference validation.

Provider calls are mocked; automated tests do not measure real-model answer accuracy.

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

## Limits and error handling

- Up to 20 questions, each at most 1,000 characters.
- Questions file: 100,000 bytes.
- Source document: 10,000,000 bytes.
- PDF: up to 200 pages.
- Extracted text: 500,000 characters.
- Index: up to 1,000 chunks.
- JSON nesting: up to 64 levels.
- Evidence: up to six passages, 6,000 characters per passage and 24,000 characters total.
- Processing deadline: 120 seconds after uploads are read.

Invalid inputs return 400 or 415; oversized inputs return 413. Processing/provider timeouts return 504, provider connection or quota failures return 503, and provider or reference-validation failures return 502.

JSON request logs include request IDs, status codes, and duration. Responses include `X-Request-ID`. Request logs exclude document content and API keys.

## Evaluation and limitations

Manually evaluated against the OneTrust SOC 2 report. Answers may omit relevant details; passage-reference validation does not guarantee factual accuracy or completeness.

Indexes are rebuilt per request. Concurrency limits apply per request, not globally. A parsing thread may continue after the processing deadline. Document content is sent to OpenAI and incurs usage costs. Authentication and persistent indexing are not implemented.