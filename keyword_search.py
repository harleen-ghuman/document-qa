import math
import re
from collections import Counter

from langchain_core.documents import Document


STOP_WORDS = set(
    "a about above after again against all am an and any are as at be "
    "because been before being below between both but by can could did "
    "do does doing down during each few for from further had has have "
    "having he her here hers herself him himself his how i if in into is "
    "it its itself just me more most my myself no nor not now of off on "
    "once only or other our ours ourselves out over own same she should "
    "so some such than that the their theirs them themselves then there "
    "these they this those through to too under until up very was we "
    "were what when where which while who whom why will with would you "
    "your yours yourself yourselves s t".split()
)

ACRONYM_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,5})s?\b")


def normalize(word: str) -> str:
    # Light plural folding so "agreements" matches "agreement".
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [
        normalize(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def expand_acronyms(
        question: str,
        documents: list[Document],
        vocabulary: set[str],
) -> list[str]:
    """Find phrases in the documents whose initials spell an acronym
    used in the question but absent from the documents themselves."""
    expansions = []

    acronyms = {
        acronym.lower()
        for acronym in ACRONYM_PATTERN.findall(question)
    }

    for acronym in acronyms:
        if acronym in STOP_WORDS or normalize(acronym) in vocabulary:
            continue

        size = len(acronym)
        for document in documents:
            words = re.findall(r"[a-z0-9]+", document.page_content.lower())
            for start in range(len(words) - size + 1):
                phrase = words[start:start + size]
                if (
                        phrase[0] not in STOP_WORDS
                        and phrase[-1] not in STOP_WORDS
                        and "".join(word[0] for word in phrase) == acronym
                ):
                    expansions.extend(tokenize(" ".join(phrase)))
                    break
            else:
                continue
            break

    return expansions


def keyword_search(
        documents: list[Document],
        question: str,
        k: int = 3,
) -> list[Document]:
    if not documents:
        return []

    tokens = [tokenize(doc.page_content) for doc in documents]
    counts = [Counter(words) for words in tokens]
    document_frequencies = Counter(
        word
        for counts_for_document in counts
        for word in counts_for_document
    )

    query = tokenize(question)
    query.extend(
        expand_acronyms(question, documents, set(document_frequencies))
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
