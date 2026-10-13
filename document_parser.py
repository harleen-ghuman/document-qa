import json
from io import BytesIO
from json_validation import validate_json_depth
from fastapi import HTTPException
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_PAGES = 200
MAX_EXTRACTED_CHARACTERS = 500_000

def parse_document(content: bytes, filename: str) -> list[Document]:
    extension = filename.lower()

    if extension.endswith(".json"):
        validate_json_depth(content, "Document")
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Document file must contain valid JSON.",
            )
        except RecursionError:
            raise HTTPException(
                status_code=400,
                detail="Document JSON is nested too deeply.",
            )

        if not isinstance(data, (dict, list)) or not data:
            raise HTTPException(
                status_code=400,
                detail="Document must contain a non-empty object or array.",
            )

        documents = [
            Document(
                page_content=json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                metadata={"source": filename},
            )
        ]

    elif extension.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted and reader.decrypt("") == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported.",
                )

            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail="PDF exceeds the 200-page processing limit.",
                )

            extracted_characters = 0

            documents = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                extracted_characters += len(text)

                if extracted_characters > MAX_EXTRACTED_CHARACTERS:
                    raise HTTPException(
                        status_code=413,
                        detail="Document exceeds the extracted-text limit.",
                    )

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_number,
                            },
                        )
                    )

        except (PdfReadError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Could not read the PDF.",
            )

    else:
        raise HTTPException(
            status_code=415,
            detail="Document must be a PDF or JSON file.",
        )
    if sum(len(doc.page_content) for doc in documents) > (
            MAX_EXTRACTED_CHARACTERS
    ):
        raise HTTPException(
            status_code=413,
            detail="Document exceeds the extracted-text limit.",
        )

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Document contains no extractable text.",
        )

    return documents