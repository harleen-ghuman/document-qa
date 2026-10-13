from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str
    page: int | None = None
    text: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    retrieved_evidence: list[Evidence] | None = None


class AnswerResponse(BaseModel):
    chunk_count: int | None = None
    results: list[QuestionAnswer]

class GroundedAnswer(BaseModel):
    supporting_passage_ids: list[int] = Field(
        max_length=6,
        description=(
            "IDs of supplied passages supporting the answer. "
            "For passages labeled P1, P2, etc., return integers 1, 2, etc. "
            "Use an empty list when no passage answers the question."
        ),
    )
    answer: str = Field(
        description=(
            "Answer only from the supporting passages. Preserve qualifications, "
            "relationships, timing conditions, and contradictions. "
            "Identify unsupported parts without inventing facts."
        ),
    )