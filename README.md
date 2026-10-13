# Zania Document QA

A document question-answering application with a FastAPI backend and React/TypeScript frontend. Supports JSON and PDF documents using LangChain, OpenAI, and an in-memory Qdrant vector database.

## Requirements

- Docker for running the complete application.
- An OpenAI API key for live requests.
- For local development: Python 3.14 and Node.js/npm.

## Configuration

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Do not commit API keys.

## Run with Docker

```bash
docker build -t zania-document-qa .
docker run --rm --name zania-qa --env-file .env -p 127.0.0.1:8001:8000 zania-document-qa
```

Open:

- http://127.0.0.1:8001/ — React interface.
- http://127.0.0.1:8001/docs — interactive API.

The image builds the React frontend and serves it alongside FastAPI. Qdrant runs in memory; no separate database server is required.

## Local development

### Backend

From the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The API is available at http://127.0.0.1:8000/docs.

### Frontend

In a separate terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite, normally http://localhost:5173. The development proxy forwards `/api` requests to FastAPI on port 8000.

## Using the interface

1. Upload a PDF or JSON source document.
2. Type one question or upload a questions JSON file.
3. Click **Get answers**.
4. Expand **View retrieved sources** to inspect evidence.

Sources are included in the same response; expanding them does not make another AI request.

## API

### GET /api/health

Returns:

```json
{"status": "UP"}
```

### POST /api/answer

Upload two multipart files:

- `questions_file`: a JSON object containing a non-empty `questions` list.
- `document_file`: a JSON or PDF document to search.

Example questions file:

```json
{
  "questions": [
    "Which cloud provider hosts the service?",
    "What is the incident notification policy?"
  ]
}
```

Example JSON document:

```json
{
  "cloud_provider": "AWS",
  "incident_notification_policy": "Notify customers without undue delay."
}
```

Example request against the Docker application:

```bash
curl -X POST http://127.0.0.1:8001/api/answer \
  -F "questions_file=@sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your document's path. For the local backend, use port 8000.

By default, the response contains ordered question-answer pairs:

```json
{
  "results": [
    {
      "question": "Which cloud provider hosts the service?",
      "answer": "The service is hosted on AWS."
    }
  ]
}
```

Set `include_evidence=true` to also return `chunk_count` and `retrieved_evidence` containing source filenames, passage text, and PDF page numbers where available:

```bash
curl -X POST "http://127.0.0.1:8001/api/answer?include_evidence=true" \
  -F "questions_file=@sample_questions.json" \
  -F "document_file=@report.pdf"
```

In `/docs`, enable `include_evidence` before executing. Null fields are omitted.

## How it works

1. Validate uploads and parse the document in a worker thread.
2. Split content into overlapping chunks.
3. Generate embeddings with `text-embedding-3-small` and build an in-memory Qdrant index.
4. Combine semantic retrieval for the original and rewritten questions with BM25-style keyword retrieval.
5. Expand PDF matches to original pages, remove duplicates, and retain up to six evidence passages.
6. Generate a structured answer with supporting quotations using `gpt-4o-mini`.

Up to three questions run concurrently per request. Results remain in input order, and outstanding question tasks are cancelled if one fails.

Supporting quotations are internal to generation. Retrieved passages are available in evidence mode and are not independently verified claim-level citations.

## Project structure

- `frontend/`: React/TypeScript upload interface and results display.
- `main.py`: HTTP routes, upload validation, frontend serving, and request logging.
- `qa_service.py`: indexing, retrieval, generation, and bounded concurrency.
- `document_parser.py`: JSON and PDF parsing.
- `keyword_search.py`: keyword ranking.
- `schemas.py`: response and internal generation models.
- `tests/`: parser, retrieval, concurrency, and mocked API tests.
- `Dockerfile`: frontend build and application packaging.

## Tests

From the project root:

```bash
python -m pytest -q
```

All 31 backend tests passed. Tests cover input validation, upload limits, PDF parsing errors, timeout handling, keyword retrieval, evidence delivery, response fields, concurrency limits, and result ordering.

Provider calls are mocked in automated tests; these tests do not measure real-model answer accuracy. Pinned Python dependencies were also verified in a fresh virtual environment.

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

Both checks passed. Docker startup, frontend serving, JSON request logging, and live JSON document answering were also verified.

## Limits and errors

- Maximum 20 questions per request.
- Maximum 1,000 characters per question.
- Questions upload limit: 100,000 bytes.
- Document upload limit: 10,000,000 bytes.
- Provider timeouts return HTTP 504.
- Provider connection or rate/quota failures return HTTP 503.
- Other provider API errors return HTTP 502.

Requests produce JSON logs with request IDs, HTTP status codes, and duration in milliseconds. Responses include an `X-Request-ID` header. Uploaded content and API keys are not included in these request logs.

## Evaluation and limitations

Manually evaluated against the OneTrust SOC 2 report, including a 19-question run. Some answers remain incomplete or overstate the evidence; structured output does not guarantee accuracy.

The index is rebuilt per request. Question concurrency is limited per request, not globally. Document content is sent to OpenAI, incurring usage costs. Authentication, persistent indexing, and an overall request deadline are not implemented.