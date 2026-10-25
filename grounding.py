import json

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from schemas import GroundedAnswer


NOT_FOUND = "Not found in the provided document."


def validate_grounding(
        response: GroundedAnswer,
        evidence: list[Document],
        question: str = "",
) -> None:
    ids = response.supporting_passage_ids
    answer = response.answer.strip()

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="AI returned an empty answer.",
        )

    if len(ids) > 6 or len(ids) != len(set(ids)):
        raise HTTPException(
            status_code=502,
            detail="AI returned invalid source references.",
        )

    if any(
            type(passage_id) is not int
            or passage_id < 1
            or passage_id > len(evidence)
            for passage_id in ids
    ):
        raise HTTPException(
            status_code=502,
            detail="AI referenced a passage that was not supplied.",
        )

    if answer != NOT_FOUND and not ids:
        raise HTTPException(
            status_code=502,
            detail="AI response did not include supporting evidence.",
        )

    if answer == NOT_FOUND and ids:
        raise HTTPException(
            status_code=502,
            detail="AI returned inconsistent answer and source references.",
        )


async def generate_verified_answer(
        model,
        messages,
        evidence: list[Document],
        question: str,
):
    response = await model.ainvoke(messages)

    try:
        validate_grounding(response, evidence, question=question)

        if response.answer.strip() != NOT_FOUND or not evidence:
            return response

    except HTTPException as error:
        if error.status_code != 502:
            raise

    reviewed = await model.ainvoke([
        *messages,
        AIMessage(
            content=json.dumps({
                "supporting_passage_ids": response.supporting_passage_ids,
                "answer": response.answer,
            })
        ),
        HumanMessage(
            content=(
                "Review the original passages again for factual support. "
                "A passage may partially answer the question. Conditions, "
                "triggering events, and qualitative terms are relevant "
                "even without exact figures. Report supported conditions "
                "and explicitly state what is unspecified. Keep distinct "
                "concepts separate, and report relevant exceptions or "
                "contradictions. "
                "Do not force an answer when support is absent. "
                f"Use unique supporting passage IDs from 1 to {len(evidence)}. "
                "Return an empty ID list and exactly "
                f"'{NOT_FOUND}' only if no passage answers any part."
            )
        ),
    ])

    validate_grounding(reviewed, evidence, question=question)
    return reviewed