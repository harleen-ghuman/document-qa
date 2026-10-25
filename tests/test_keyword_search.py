from langchain_core.documents import Document

from keyword_search import keyword_search


def test_empty_documents_return_no_matches():
    assert keyword_search([], "cloud provider") == []


def test_unrelated_question_returns_no_matches():
    documents = [
        Document(page_content="The service is hosted on AWS.")
    ]

    assert keyword_search(documents, "temperature humidity") == []


def test_relevant_document_ranks_first_and_respects_limit():
    relevant = Document(page_content="Incident notification policy.")
    unrelated = Document(page_content="Employee vacation policy.")

    matches = keyword_search(
        [unrelated, relevant],
        "incident notification",
        k=1,
    )

    assert matches == [relevant]


def test_acronym_expansion_finds_full_phrase():
    relevant = Document(
        page_content="The service level agreement defines notification."
    )
    unrelated = Document(page_content="The office opens on Monday.")

    matches = keyword_search(
        [unrelated, relevant],
        "SLA",
        k=1,
    )

    assert matches == [relevant]


def test_acronym_expansion_works_for_any_acronym():
    relevant = Document(
        page_content="Our recovery point objective is four hours."
    )
    unrelated = Document(page_content="Lunch is served at noon.")

    matches = keyword_search(
        [unrelated, relevant],
        "What is the RPO?",
        k=1,
    )

    assert matches == [relevant]


def test_acronym_present_in_document_is_not_expanded():
    with_acronym = Document(page_content="The API returns JSON.")
    phrase_only = Document(page_content="An apple pie is ready.")

    matches = keyword_search(
        [phrase_only, with_acronym],
        "API",
        k=2,
    )

    assert matches == [with_acronym]


def test_plural_and_singular_forms_match():
    relevant = Document(page_content="Signed agreements are archived.")
    unrelated = Document(page_content="The office opens on Monday.")

    matches = keyword_search(
        [unrelated, relevant],
        "agreement",
        k=1,
    )

    assert matches == [relevant]


def test_search_preserves_source_and_page_metadata():
    document = Document(
        page_content="Customer data is encrypted.",
        metadata={"source": "report.pdf", "page": 7},
    )

    matches = keyword_search([document], "encrypted")

    assert len(matches) == 1
    assert matches[0].metadata == {
        "source": "report.pdf",
        "page": 7,
    }
    assert matches[0].page_content == "Customer data is encrypted."