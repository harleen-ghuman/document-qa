import math
import re
from collections import Counter

from langchain_core.documents import Document


STOP_WORDS = set(
    "the a an is are of to on in and what which does "
    "how long company s its".split()
)


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def keyword_search(
        documents: list[Document],
        question: str,
        k: int = 3,
) -> list[Document]:
    if not documents:
        return []

    query = tokenize(question)

    if "sla" in query or "slas" in query:
        query.extend(["service", "level", "agreement", "agreements"])

    tokens = [tokenize(doc.page_content) for doc in documents]
    counts = [Counter(words) for words in tokens]
    document_frequencies = Counter(
        word
        for counts_for_document in counts
        for word in counts_for_document
    )
    average_length = sum(map(len, tokens)) / len(tokens) or 1
    scores = []

    for index, counts_for_document in enumerate(counts):
        score = 0.0

        for word in query:
            frequency = counts_for_document[word]
            document_frequency = document_frequencies[word]

            inverse_frequency = math.log(
                1 + (
                        len(documents) - document_frequency + 0.5
                ) / (document_frequency + 0.5)
            )

            denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(tokens[index]) / average_length
            )

            score += (
                    inverse_frequency * frequency * 2.5 / denominator
            )

        scores.append(score)

    ranked = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        documents[index]
        for index in ranked[:k]
        if scores[index] > 0
    ]