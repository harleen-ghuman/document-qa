import json
from io import BytesIO
import document_parser
import pytest
from fastapi import HTTPException
from pypdf import PdfWriter

from document_parser import parse_document


@pytest.mark.parametrize(
    "data",
    [
        {"cloud_provider": "AWS", "name": "Café"},
        [{"cloud_provider": "AWS"}],
    ],
)
def test_json_preserves_content_and_source(data):
    documents = parse_document(
        json.dumps(data).encode(),
        "document.json",
    )

    assert len(documents) == 1
    assert json.loads(documents[0].page_content) == data
    assert documents[0].metadata == {"source": "document.json"}


@pytest.mark.parametrize(
    "content",
    [b"not JSON", b"\xff"],
)
def test_invalid_json_returns_400(content):
    with pytest.raises(HTTPException) as error:
        parse_document(content, "document.json")

    assert error.value.status_code == 400
    assert error.value.detail == "Document file must contain valid JSON."


@pytest.mark.parametrize(
    "content",
    [b"{}", b"[]", b"null", b'"text"', b"123"],
)
def test_empty_or_scalar_json_returns_400(content):
    with pytest.raises(HTTPException) as error:
        parse_document(content, "document.json")

    assert error.value.status_code == 400
    assert error.value.detail == (
        "Document must contain a non-empty object or array."
    )


def test_unsupported_file_returns_415():
    with pytest.raises(HTTPException) as error:
        parse_document(b"hello", "document.txt")

    assert error.value.status_code == 415


def test_corrupt_pdf_returns_400():
    with pytest.raises(HTTPException) as error:
        parse_document(b"not a PDF", "document.pdf")

    assert error.value.status_code == 400
    assert error.value.detail == "Could not read the PDF."


def test_blank_pdf_returns_400():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(HTTPException) as error:
        parse_document(buffer.getvalue(), "document.pdf")

    assert error.value.status_code == 400
    assert error.value.detail == "Document contains no extractable text."


def test_password_protected_pdf_returns_400():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("test-password")
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(HTTPException) as error:
        parse_document(buffer.getvalue(), "document.pdf")

    assert error.value.status_code == 400
    assert error.value.detail == (
        "Password-protected PDFs are not supported."
    )

def test_deeply_nested_json_returns_400():
    content = b"[" * 2000 + b"0" + b"]" * 2000

    with pytest.raises(HTTPException) as error:
        parse_document(content, "document.json")

    assert error.value.status_code == 400
    assert error.value.detail == "Document JSON is nested too deeply."


def test_json_preserves_null_and_false_values():
    data = {
        "provider": "AWS",
        "retention_days": None,
        "enabled": False,
        "count": 0,
    }

    documents = parse_document(
        json.dumps(data).encode(),
        "document.json",
    )

    assert json.loads(documents[0].page_content) == data

def test_pdf_page_limit_returns_413(monkeypatch):
    monkeypatch.setattr(document_parser, "MAX_PDF_PAGES", 1)

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)

    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(HTTPException) as error:
        parse_document(buffer.getvalue(), "document.pdf")

    assert error.value.status_code == 413
    assert error.value.detail == (
        "PDF exceeds the 200-page processing limit."
    )


def test_json_extracted_text_limit_returns_413(monkeypatch):
    monkeypatch.setattr(
        document_parser,
        "MAX_EXTRACTED_CHARACTERS",
        20,
    )

    content = json.dumps({"text": "a" * 100}).encode()

    with pytest.raises(HTTPException) as error:
        parse_document(content, "document.json")

    assert error.value.status_code == 413
    assert error.value.detail == (
        "Document exceeds the extracted-text limit."
    )