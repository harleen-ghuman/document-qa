import asyncio

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from grounding import (
    NOT_FOUND,
    generate_verified_answer,
    validate_grounding,
)
from schemas import GroundedAnswer


EVIDENCE = [Document(page_content="The company uses AWS.")]


def test_valid_reference_is_accepted():
    validate_grounding(
        GroundedAnswer(
            supporting_passage_ids=[1],
            answer="The company uses AWS.",
        ),
        EVIDENCE,
    )


def test_unknown_reference_is_rejected():
    with pytest.raises(HTTPException) as error:
        validate_grounding(
            GroundedAnswer(
                supporting_passage_ids=[2],
                answer="The company uses AWS.",
            ),
            EVIDENCE,
        )

    assert error.value.status_code == 502


def test_answer_without_support_is_rejected():
    with pytest.raises(HTTPException) as error:
        validate_grounding(
            GroundedAnswer(
                supporting_passage_ids=[],
                answer="The company uses AWS.",
            ),
            EVIDENCE,
        )

    assert error.value.status_code == 502


def test_not_found_without_references_is_allowed():
    validate_grounding(
        GroundedAnswer(
            supporting_passage_ids=[],
            answer=NOT_FOUND,
        ),
        [],
    )


def test_reference_correction_is_bounded():
    async def check():
        class FakeModel:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                return next(self.responses)

        invalid = GroundedAnswer(
            supporting_passage_ids=[99],
            answer="The company uses AWS.",
        )
        valid = GroundedAnswer(
            supporting_passage_ids=[1],
            answer="The company uses AWS.",
        )

        model = FakeModel([invalid, valid])
        result = await generate_verified_answer(
            model, [], EVIDENCE, "Which provider?"
        )

        assert result == valid
        assert model.calls == 2

        failing_model = FakeModel([invalid, invalid])

        with pytest.raises(HTTPException) as error:
            await generate_verified_answer(
                failing_model, [], EVIDENCE, "Which provider?"
            )

        assert error.value.status_code == 502
        assert failing_model.calls == 2

    asyncio.run(check())

def test_abstention_receives_one_answerability_review():
    async def check():
        class FakeModel:
            calls = 0

            async def ainvoke(self, messages):
                self.calls += 1

                if self.calls == 1:
                    return GroundedAnswer(
                        supporting_passage_ids=[],
                        answer=NOT_FOUND,
                    )

                return GroundedAnswer(
                    supporting_passage_ids=[1],
                    answer="The company uses AWS.",
                )

        model = FakeModel()

        result = await generate_verified_answer(
            model, [], EVIDENCE, "Which provider?"
        )

        assert result.supporting_passage_ids == [1]
        assert model.calls == 2

    asyncio.run(check())