# Zania Document QA

A FastAPI service that answers uploaded questions using evidence from an
uploaded document.

## Requirements

- Python 3.14 — the version used during development
- An OpenAI API key for live requests

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run

```bash
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

## API

### GET /api/health

Returns:

```json
{"status": "UP"}
```

### POST /api/answer

Upload two multipart files:

- `questions_file`: a JSON object containing a non-empty `questions` list.
- `document_file`: the document to search.

Example questions file:

```json
{
  "questions": [
    "Which cloud provider hosts the service?",
    "What is the incident notification policy?"
  ]
}
```

Example request using the supplied PDF:

```bash
curl -X POST http://127.0.0.1:8000/api/answer \
  -F "questions_file=@sample_questions.json" \
  -F "document_file=@report.pdf"
```

Replace `report.pdf` with your document's path.

The response contains `chunk_count` and ordered `results`. Each result
includes the original question, an answer, and retrieved evidence with
source, page, and text.

## How it works

1. Validate the uploads and parse the document.
2. Split content into overlapping chunks.
3. Embed chunks with OpenAI and build an in-memory Qdrant index.
4. Combine semantic retrieval for the original and rewritten questions
   with BM25-style keyword retrieval.
5. Expand PDF matches to full pages, remove duplicates, and retain up
   to six evidence passages.
6. Generate a structured answer with supporting quotations.

Supporting quotations are internal to generation. The API returns the
answer and retrieved passages; those passages are retrieval evidence,
not a guarantee that every answer claim is supported.

## Project structure

- `main.py`: HTTP routes, upload validation, and provider error handling.
- `qa_service.py`: indexing, retrieval, question rewriting, and generation.
- `document_parser.py`: document parsing.
- `keyword_search.py`: keyword ranking.
- `schemas.py`: response and internal generation schemas.
- `tests/test_api.py`: API validation and mocked QA-flow tests.

## Tests

```bash
python -m pytest -q
```

The current suite has 11 tests. AI clients are mocked in the tested
QA paths, so these tests do not evaluate real-model answer quality.

## Limits and errors

- Maximum 20 questions per request.
- Maximum 1,000 characters per question.
- Questions upload limit: 100,000 bytes.
- Document upload limit: 10,000,000 bytes.
- Provider timeouts return HTTP 504.
- Provider connection or rate/quota failures return HTTP 503.
- Other provider API errors return HTTP 502.

## Evaluation and known limitations

A manual evaluation used four questions against the supplied OneTrust
SOC 2 report. The latest reviewed run improved the hosting-provider
distinction and reported notification timing without inventing a
numeric deadline.

Two answers remained incomplete:

- Retention omitted the service-termination destruction condition.
- Regions omitted an additional location and inconsistent report wording.

These are partial answers, not a perfect evaluation result. Structured
output enforces response format; it does not guarantee factual accuracy
or completeness.

The service rebuilds its index for each request and processes questions
sequentially. Live requests incur OpenAI usage costs. Uploaded content
is sent to OpenAI for embedding, and selected evidence is sent for answer
generation.

This assessment implementation does not include authentication,
persistent indexing, or an overall request deadline.
