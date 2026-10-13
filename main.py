import json
import logging
import time
import uuid
import asyncio
from langchain_core.callbacks import get_usage_metadata_callback
from json_validation import validate_json_depth
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from schemas import AnswerResponse
from document_parser import parse_document
from qa_service import answer_questions
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


load_dotenv()
app = FastAPI(title="Zania Document QA")

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(FRONTEND_DIST / "index.html")

request_logger = logging.getLogger("zania.requests")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False

if not request_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    status_code = 500

    with get_usage_metadata_callback() as usage:
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            token_usage = list(usage.usage_metadata.values())

            request_logger.info(
                json.dumps({
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                        ),
                    "prompt_tokens": sum(
                        item.get("input_tokens", 0)
                        for item in token_usage
                    ),
                    "completion_tokens": sum(
                        item.get("output_tokens", 0)
                        for item in token_usage
                    ),
                    "total_tokens": sum(
                        item.get("total_tokens", 0)
                        for item in token_usage
                    ),
                })
            )

MAX_QUESTIONS = 20
MAX_QUESTION_LENGTH = 1000
MAX_QUESTIONS_BYTES = 100_000
MAX_DOCUMENT_BYTES = 10_000_000
PROCESSING_TIMEOUT_SECONDS = 120


@app.exception_handler(APITimeoutError)
async def handle_openai_timeout(request: Request, exc: APITimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": "AI provider timed out. Please try again."},
    )


@app.exception_handler(RateLimitError)
async def handle_openai_rate_limit(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "AI provider is temporarily unavailable due to "
                      "rate or quota limits."
        },
    )


@app.exception_handler(APIConnectionError)
async def handle_openai_connection(request: Request, exc: APIConnectionError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Could not connect to the AI provider."},
    )


@app.exception_handler(APIStatusError)
async def handle_openai_status(request: Request, exc: APIStatusError):
    return JSONResponse(
        status_code=502,
        content={"detail": "AI provider could not complete the request."},
    )


@app.get("/api/health")
def health():
    return {"status": "UP"}


@app.post("/api/answer", response_model=AnswerResponse, response_model_exclude_none=True,)
async def answer(
        questions_file: UploadFile = File(...),
        document_file: UploadFile = File(...),
        include_evidence: bool = False,
):
    content = await questions_file.read(MAX_QUESTIONS_BYTES + 1)

    if len(content) > MAX_QUESTIONS_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Questions file exceeds the 100 KB limit.",
        )

    validate_json_depth(content, "Questions")

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400,
            detail="Questions file must contain valid JSON.",
        )
    except RecursionError:
        raise HTTPException(
            status_code=400,
            detail="Questions JSON is nested too deeply.",
        )

    if isinstance(data, dict):
        questions = data.get("questions")
    elif isinstance(data, list):
        questions = data
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Questions file must contain a JSON object "
                "with 'questions' or a JSON list."
            ),
        )

    if not isinstance(questions, list) or not questions:
        raise HTTPException(
            status_code=400,
            detail="'questions' must be a non-empty list.",
        )

    if any(
            not isinstance(question, str) or not question.strip()
            for question in questions
    ):
        raise HTTPException(
            status_code=400,
            detail="Each question must be a non-empty string.",
        )
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail="A maximum of 20 questions is allowed.",
        )

    if any(len(question) > MAX_QUESTION_LENGTH for question in questions):
        raise HTTPException(
            status_code=400,
            detail="Each question must be at most 1000 characters.",
        )
    document_content = await document_file.read(MAX_DOCUMENT_BYTES + 1)

    if len(document_content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Document exceeds the 10 MB limit.",
        )

    try:
        async with asyncio.timeout(PROCESSING_TIMEOUT_SECONDS):
            documents = await asyncio.to_thread(
                parse_document,
                content=document_content,
                filename=document_file.filename or "",
            )

            result = await answer_questions(
                documents=documents,
                questions=questions,
                source_name=document_file.filename or "uploaded-document",
            )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Document processing timed out. "
                "Try a smaller document or fewer questions."
            ),
        )

    if not include_evidence:
        result.pop("chunk_count", None)
        for item in result["results"]:
            item.pop("retrieved_evidence", None)

    return result



