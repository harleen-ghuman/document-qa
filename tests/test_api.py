import json
import qa_service
import main
import httpx
from openai import APITimeoutError
from fastapi.testclient import TestClient
from types import SimpleNamespace
from langchain_core.documents import Document

client = TestClient(main.app)


def upload(
        questions_content,
        document_content=b'{"cloud_provider":"AWS"}',
        include_evidence=True,
):
    return client.post(
        "/api/answer",
        params={"include_evidence": include_evidence},
        files={
            "questions_file": (
                "questions.json",
                questions_content,
                "application/json",
            ),
            "document_file": (
                "document.json",
                document_content,
                "application/json",
            ),
        },
    )


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_missing_uploads():
    response = client.post("/api/answer")
    assert response.status_code == 422


def test_invalid_questions_json():
    response = upload(b"this is not JSON")
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Questions file must contain valid JSON."
    )


def test_empty_questions():
    response = upload(json.dumps({"questions": []}).encode())
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "'questions' must be a non-empty list."
    )


def test_too_many_questions():
    response = upload(
        json.dumps({"questions": ["Question?"] * 21}).encode()
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "A maximum of 20 questions is allowed."
    )


def test_question_too_long():
    response = upload(
        json.dumps({"questions": ["a" * 1001]}).encode()
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Each question must be at most 1000 characters."
    )


def test_questions_file_too_large(monkeypatch):
    monkeypatch.setattr(main, "MAX_QUESTIONS_BYTES", 10)
    response = upload(b"x" * 11)
    assert response.status_code == 413


def test_document_file_too_large(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOCUMENT_BYTES", 10)
    response = upload(
        json.dumps({"questions": ["Which provider?"]}).encode(),
        document_content=b"x" * 11,
    )
    assert response.status_code == 413

def test_embedding_timeout_returns_504(monkeypatch):
    # Replace the real embedding client.
    monkeypatch.setattr(
        "qa_service.OpenAIEmbeddings",
        lambda **kwargs: object(),
    )

    # Simulate a timeout while building the document index.
    async def fake_afrom_texts(*args, **kwargs):
        raise APITimeoutError(
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/embeddings",
            )
        )

    monkeypatch.setattr(
        qa_service.QdrantVectorStore,
        "afrom_texts",
        fake_afrom_texts,
    )

    response = upload(
        json.dumps({"questions": ["Which provider?"]}).encode()
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "AI provider timed out. Please try again."
    )

def test_answers_every_question_in_order(monkeypatch):
    monkeypatch.setenv("ENABLE_QUERY_REWRITE", "true")
    async def fake_rewrite_question(question, model):
        return question

    monkeypatch.setattr(
        qa_service,
        "rewrite_question",
        fake_rewrite_question,
    )

    monkeypatch.setattr(
        "qa_service.OpenAIEmbeddings",
        lambda **kwargs: object(),
    )

    retrieved_questions = []
    model_calls = []

    class FakeVectorStore:
        async def asimilarity_search(self, question, k):
            retrieved_questions.append(question)
            return [
                Document(page_content="The company uses AWS.")
            ]

    async def fake_afrom_texts(*args, **kwargs):
        return FakeVectorStore()

    monkeypatch.setattr(
        qa_service.QdrantVectorStore,
        "afrom_texts",
        fake_afrom_texts,
    )

    class FakeChatModel:
        def with_structured_output(self, *args, **kwargs):
            return self
        async def ainvoke(self, messages):
            model_calls.append(messages)
            return SimpleNamespace(
                supporting_passage_ids=[1],
                answer=f"Answer {len(model_calls)}"
            )

    monkeypatch.setattr(
        "qa_service.ChatOpenAI",
        lambda **kwargs: FakeChatModel(),
    )

    questions = [
        "Which provider?",
        "What is the SLA?",
    ]

    monkeypatch.setattr(
        qa_service,
        "keyword_search",
        lambda *args, **kwargs: [],
    )

    response = upload(
        json.dumps({"questions": questions}).encode()
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert retrieved_questions == questions
    assert len(model_calls) == 2
    assert [
               (result["question"], result["answer"])
               for result in results
           ] == [
               ("Which provider?", "Answer 1"),
               ("What is the SLA?", "Answer 2"),
           ]

    for result in results:
        assert result["retrieved_evidence"] == [
            {
                "source": "document.json",
                "text": "The company uses AWS.",
            }
        ]

        for messages in model_calls:
            assert "The company uses AWS." in messages[1].content

    concise_response = upload(
        json.dumps(questions).encode(),
        include_evidence=False,
    )

    assert concise_response.status_code == 200
    concise_body = concise_response.json()
    assert "chunk_count" not in concise_body
    assert len(concise_body["results"]) == len(questions)

    for item in concise_body["results"]:
        assert set(item) == {"question", "answer", "citations"}
        assert item["citations"] == [{"source": "document.json"}]

def test_keyword_evidence_reaches_model(monkeypatch):
    async def fake_rewrite(question, model):
        return question

    monkeypatch.setattr(qa_service, "rewrite_question", fake_rewrite)
    monkeypatch.setattr(
        "qa_service.OpenAIEmbeddings",
        lambda **kwargs: object(),
    )

    class FakeStore:
        async def asimilarity_search(self, question, k):
            return []

    async def fake_index(*args, **kwargs):
        return FakeStore()

    monkeypatch.setattr(
        qa_service.QdrantVectorStore,
        "afrom_texts",
        fake_index,
    )

    monkeypatch.setattr(
        qa_service,
        "keyword_search",
        lambda *args, **kwargs: [
            Document(
                page_content="Notify customers without undue delay.",
                metadata={"source": "document.json"},
            )
        ],
    )

    class FakeModel:
        def with_structured_output(self, *args, **kwargs):
            return self
        async def ainvoke(self, messages):
            assert "Notify customers without undue delay." in (
                messages[1].content
            )
            return SimpleNamespace(
                supporting_passage_ids=[1],
                answer="Without undue delay.",)

    monkeypatch.setattr(
        "qa_service.ChatOpenAI",
        lambda **kwargs: FakeModel(),
    )

    response = upload(
        json.dumps({"questions": ["Notification timing?"]}).encode()
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["answer"] == "Without undue delay."
    assert result["retrieved_evidence"][0]["text"] == (
        "Notify customers without undue delay."
    )

def test_deeply_nested_questions_json_returns_400():
    content = b"[" * 2000 + b"0" + b"]" * 2000

    response = upload(content)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Questions JSON is nested too deeply."
    )

def test_chunk_limit_rejects_before_embedding(monkeypatch):
    monkeypatch.setattr(qa_service, "MAX_CHUNKS", 1)

    class FakeSplitter:
        def split_documents(self, documents):
            return [
                Document(page_content="First chunk."),
                Document(page_content="Second chunk."),
            ]

    monkeypatch.setattr(
        qa_service,
        "RecursiveCharacterTextSplitter",
        lambda **kwargs: FakeSplitter(),
    )

    def unexpected_embeddings(**kwargs):
        raise AssertionError(
            "Embedding client must not be created for an oversized document."
        )

    monkeypatch.setattr(
        qa_service,
        "OpenAIEmbeddings",
        unexpected_embeddings,
    )

    response = upload(
        json.dumps({"questions": ["Which provider?"]}).encode()
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Document exceeds the 1000-chunk processing limit."
    )

def test_processing_deadline_returns_504(monkeypatch):
    import asyncio

    monkeypatch.setattr(main, "PROCESSING_TIMEOUT_SECONDS", 0.1)

    async def stalled_answer(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(main, "answer_questions", stalled_answer)

    response = upload(
        json.dumps({"questions": ["Which provider?"]}).encode()
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "Document processing timed out. "
        "Try a smaller document or fewer questions."
    )