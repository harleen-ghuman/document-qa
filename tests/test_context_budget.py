from langchain_core.documents import Document

import qa_service


def test_evidence_respects_total_and_passage_limits(monkeypatch):
    monkeypatch.setattr(qa_service, "MAX_CONTEXT_CHARACTERS", 100)
    monkeypatch.setattr(qa_service, "MAX_PASSAGE_CHARACTERS", 60)

    matches = [
        Document(page_content="a" * 200),
        Document(page_content="b" * 200),
        Document(page_content="c" * 200),
    ]

    bounded = qa_service.limit_evidence(matches)
    context = "\n\n".join(doc.page_content for doc in bounded)

    assert len(context) <= 100
    assert all(len(doc.page_content) <= 60 for doc in bounded)
    assert len(bounded) == 2
    assert "[Passage truncated.]" in context


def test_evidence_preserves_metadata_without_mutating_source(monkeypatch):
    monkeypatch.setattr(qa_service, "MAX_PASSAGE_CHARACTERS", 60)

    original = Document(
        page_content="a" * 200,
        metadata={"source": "report.pdf", "page": 7},
    )

    bounded = qa_service.limit_evidence([original])

    assert bounded[0].metadata == original.metadata
    assert original.page_content == "a" * 200
    assert len(bounded[0].page_content) <= 60


def test_empty_passages_are_skipped():
    matches = [
        Document(page_content=" \n "),
        Document(page_content="Relevant evidence."),
    ]

    bounded = qa_service.limit_evidence(matches)

    assert len(bounded) == 1
    assert bounded[0].page_content == "Relevant evidence."