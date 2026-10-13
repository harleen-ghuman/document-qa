import sys
from pathlib import Path

from document_parser import parse_document

path = Path(sys.argv[1])

documents = parse_document(
    content=path.read_bytes(),
    filename=path.name,
)

page_12 = [
    document
    for document in documents
    if document.metadata.get("page") == 12
]

print("Extracted pages:", len(documents))
print("Page 12 found:", bool(page_12))

for document in page_12:
    print(document.page_content)