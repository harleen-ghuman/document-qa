# Zania Document QA

A FastAPI service that answers uploaded questions using evidence from a JSON or PDF document. Built with LangChain, OpenAI, and Qdrant.

## Requirements

- Python 3.14 — the version used for development and testing.
- An OpenAI API key for live requests.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Do not commit API keys.

## Run

```bash
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

Qdrant runs in memory; no separate database server is required.

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

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/answer \
  -F "questions_file=@sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your JSON or PDF document's path.

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

Set `include_evidence=true` to also return `chunk_count` and retrieved passages with source filenames and PDF page numbers where available:

```bash
curl -X POST "http://127.0.0.1:8000/api/answer?include_evidence=true" \
  -F "questions_file=@sample_questions.json" \
  -F "document_file=@report.pdf"
```

In `/docs`, enable the `include_evidence` parameter before executing.

## How it works

1. Validate uploads and parse the document.
2. Split content into overlapping chunks.
3. Generate embeddings with `text-embedding-3-small` and build an in-memory Qdrant index.
4. Combine semantic retrieval for the original and rewritten questions with BM25-style keyword retrieval.
5. Expand PDF matches to original pages, remove duplicates, and retain up to six evidence passages.
6. Generate a structured answer with supporting quotations using `gpt-4o-mini`.

Supporting quotations are internal to generation. Retrieved passages are available in evidence mode and are not independently verified claim-level citations.

## Project structure

- `main.py`: HTTP routes, upload validation, and provider error handling.
- `qa_service.py`: indexing, retrieval, question rewriting, and generation.
- `document_parser.py`: JSON and PDF parsing.
- `keyword_search.py`: keyword ranking.
- `schemas.py`: response and internal generation models.
- `tests/test_api.py`: API validation and mocked QA-flow tests.

## Tests

```bash
python -m pytest -q
```

All 11 tests passed, including after installing pinned dependencies in a fresh virtual environment on macOS with Python 3.14.

Tests cover input validation, upload limits, timeout handling, ordered answers, keyword evidence reaching the model, and concise response fields. AI clients are mocked; these tests do not measure real-model answer accuracy.

## Limits and errors

- Maximum 20 questions per request.
- Maximum 1,000 characters per question.
- Questions upload limit: 100,000 bytes.
- Document upload limit: 10,000,000 bytes.
- Provider timeouts return HTTP 504.
- Provider connection or rate/quota failures return HTTP 503.
- Other provider API errors return HTTP 502.

## Evaluation and limitations

Manually evaluated against the OneTrust SOC 2 report, including a 19-question run. Some answers remain incomplete or overstate the evidence; structured output does not guarantee accuracy.

The index is rebuilt per request, and questions are processed sequentially. Document content is sent to OpenAI, incurring usage costs. Authentication, persistent indexing, and an overall request deadline are not implemented.