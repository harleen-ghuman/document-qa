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
    supporting_quotes: list[str] = Field(
        description=(
            "Up to three short exact quotations from the supplied passages "
            "that answer the question wholly or partially. Include relevant "
            "policies or event-based timing even without a numeric duration. "
            "Use an empty list only when no relevant support exists."
        )
    )
    answer: str = Field(
        description=(
            "Answer supported by those quotations. State what is unspecified. "
            "Distinguish a missing numeric duration from missing policy "
            "information. Do not invent facts."
        )
    )