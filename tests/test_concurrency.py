import asyncio

import qa_service


def test_question_concurrency_is_limited(monkeypatch):
    monkeypatch.setattr(qa_service, "MAX_CONCURRENT_QUESTIONS", 3)

    async def check():
        release = asyncio.Event()
        three_started = asyncio.Event()
        active = 0
        maximum_active = 0
        started = []

        async def answer_one(question):
            nonlocal active, maximum_active

            active += 1
            maximum_active = max(maximum_active, active)
            started.append(question)

            if len(started) == 3:
                three_started.set()

            try:
                await release.wait()
                return question
            finally:
                active -= 1

        questions = list(range(6))
        task = asyncio.create_task(
            qa_service.run_questions_bounded(questions, answer_one)
        )

        try:
            await asyncio.wait_for(three_started.wait(), timeout=2)
            assert len(started) == 3
            assert active == 3
        finally:
            release.set()
            results = await asyncio.wait_for(task, timeout=2)

        assert maximum_active == 3
        assert results == questions

    asyncio.run(check())


def test_results_preserve_order_when_completion_order_differs(monkeypatch):
    monkeypatch.setattr(qa_service, "MAX_CONCURRENT_QUESTIONS", 3)

    async def check():
        releases = [asyncio.Event() for _ in range(3)]
        completed = [asyncio.Event() for _ in range(3)]
        all_started = asyncio.Event()
        started = 0
        completion_order = []

        async def answer_one(question):
            nonlocal started

            started += 1
            if started == 3:
                all_started.set()

            await releases[question].wait()
            completion_order.append(question)
            completed[question].set()
            return f"Answer {question}"

        task = asyncio.create_task(
            qa_service.run_questions_bounded([0, 1, 2], answer_one)
        )

        try:
            await asyncio.wait_for(all_started.wait(), timeout=2)

            for question in [2, 1, 0]:
                releases[question].set()
                await asyncio.wait_for(
                    completed[question].wait(),
                    timeout=2,
                )
        finally:
            for release in releases:
                release.set()
            results = await asyncio.wait_for(task, timeout=2)

        assert completion_order == [2, 1, 0]
        assert results == ["Answer 0", "Answer 1", "Answer 2"]

    asyncio.run(check())