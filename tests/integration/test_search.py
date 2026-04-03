"""
RAG Evaluation Harness

End-to-end evaluation of search quality and answer accuracy against a known
corpus. Each test case specifies a natural-language question, the expected
page(s) where the answer lives, and the expected answer content.

**Retrieval quality** is graded by where the expected page lands in the
ranked results:
    EXCELLENT  rank 1-3
    GOOD       rank 4-5
    FAIR       rank 6-10
    POOR       rank > 10
    FAILED     expected page not found in top-k

**Answer accuracy** is checked by an LLM judge that compares the generated
answer against the expected answer, with a simple text-heuristic fallback.

Requirements:
    - Postgres and Meilisearch running with an ingested corpus
    - OPENAI_API_KEY set in .env or environment
    - Cross-encoder model downloaded (happens on first run)

Run:
    uv run pytest tests/integration/test_search.py -v
    uv run pytest tests/integration/test_search.py -v -k "morgan"
    uv run pytest tests/integration/test_search.py -v --tb=short
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import openai
import pytest
from dotenv import load_dotenv

from tome.search.search import search_codex  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test question definitions
# ---------------------------------------------------------------------------


@dataclass
class TestQuestion:
    """A question with expected retrieval pages and expected answer."""

    question: str
    expected_pages: list[int]
    expected_answer: str
    description: str


EVAL_QUESTIONS: list[TestQuestion] = [
    TestQuestion(
        question="When did the McDonald military expedition cross Guernsey County?",
        expected_pages=[15],
        expected_answer="1774",
        description="Military expedition timing",
    ),
    TestQuestion(
        question=(
            "What had the men involved in the Wills Creek Incident done "
            "that caused the Indians to come after them?"
        ),
        expected_pages=[55],
        expected_answer="They stole 15 horses from the Indians.",
        description="Wills Creek Incident cause",
    ),
    TestQuestion(
        question=(
            "In what years did the early settlers from the Isle of Guernsey "
            "arrive in Guernsey County?"
        ),
        expected_pages=[29],
        expected_answer="1806 and 1807",
        description="Guernsey settlers arrival years",
    ),
    TestQuestion(
        question="Did the Naftal family arrive in Cambridge in 1806?",
        expected_pages=[29],
        expected_answer="No.",
        description="Naftal family arrival verification",
    ),
    TestQuestion(
        question="What was the name of the company that built Cambridge's steel mill?",
        expected_pages=[50],
        expected_answer="The Cambridge Iron and Steel Company",
        description="Steel mill company identification",
    ),
    TestQuestion(
        question=(
            "When did Cambridge get its second major railroad going north and south?"
        ),
        expected_pages=[43],
        expected_answer="1873",
        description="Cambridge railroad expansion",
    ),
    TestQuestion(
        question="What was the old name for Maysville, Kentucky, in the 1700s?",
        expected_pages=[19],
        expected_answer="Limestone",
        description="Maysville historical name",
    ),
    TestQuestion(
        question="When did Morgan's raid reach Cumberland?",
        expected_pages=[45],
        expected_answer="July 23, 1863",
        description="Morgan's raid timing",
    ),
    TestQuestion(
        question="What can you tell me about an army hospital built near Cambridge?",
        expected_pages=[51],
        expected_answer="the Fletcher General Hospital story",
        description="Army hospital information",
    ),
    TestQuestion(
        question="When and where did glass manufacturing start in Guernsey County?",
        expected_pages=[50],
        expected_answer="1884, in Quaker City",
        description="Glass manufacturing history",
    ),
    TestQuestion(
        question="When did Cambridge's steel mill go out of business?",
        expected_pages=[50],
        expected_answer="in the 1940s",
        description="Steel mill closure",
    ),
    TestQuestion(
        question="Who were the main historians of Guernsey County?",
        expected_pages=[9],
        expected_answer="main historians of Guernsey County",
        description="County historians identification",
    ),
    TestQuestion(
        question="When did Congress authorize building the National Road?",
        expected_pages=[38],
        expected_answer="1802",
        description="National Road authorization",
    ),
    TestQuestion(
        question=(
            "Did any of the founders of Cambridge participate in the Revolutionary War?"
        ),
        expected_pages=[27],
        expected_answer="Jacob Gomber did",
        description="Cambridge founders Revolutionary War participation",
    ),
    TestQuestion(
        question="Who was the first sitting president to pass through Cambridge?",
        expected_pages=[58],
        expected_answer="James Monroe",
        description="First president to visit Cambridge",
    ),
    TestQuestion(
        question="Who built the Colonial Theater, and when?",
        expected_pages=[92],
        expected_answer="not clear who built it, but it came to be around 1901",
        description="Colonial Theater construction",
    ),
    TestQuestion(
        question="Was John Glenn ever in combat?",
        expected_pages=[51],
        expected_answer="Yes.",
        description="John Glenn combat experience",
    ),
    TestQuestion(
        question="What sort of things did Morgan's Raiders steal?",
        expected_pages=[62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
        expected_answer="lots of things, but especially horses",
        description="Morgan's Raiders stolen items",
    ),
    TestQuestion(
        question="Whose farm did the army take over to build Fletcher General Hospital?",
        expected_pages=[],
        expected_answer=(
            "not mentioned in the book; other books may mention it was the Oldham farm"
        ),
        description="Fletcher General Hospital farm ownership (not in corpus)",
    ),
    TestQuestion(
        question="Did Morgan's Raid pass through Byesville?",
        expected_pages=[],
        expected_answer="not on route mentioned in the book; No.",
        description="Morgan's Raid route through Byesville (not in corpus)",
    ),
]


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------


def _simple_answer_comparison(expected: str, actual: str) -> dict[str, Any]:
    """Heuristic fallback when the LLM judge is unavailable."""
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    if expected_lower in actual_lower:
        return {
            "match": True,
            "confidence": "medium",
            "reason": "Key information found in actual answer",
        }

    not_available_expected = (
        "not mentioned" in expected_lower or "not in book" in expected_lower
    )
    not_available_actual = any(
        p in actual_lower
        for p in ["not provide", "does not", "no information", "not mention"]
    )
    if not_available_expected and not_available_actual:
        return {
            "match": True,
            "confidence": "medium",
            "reason": "Both indicate information not available",
        }

    if expected_lower.strip() in ("no.", "no") and any(
        p in actual_lower for p in ["no", "did not", "was not"]
    ):
        return {
            "match": True,
            "confidence": "medium",
            "reason": "Both indicate negative answer",
        }

    return {
        "match": False,
        "confidence": "low",
        "reason": "Simple comparison found no match",
    }


def compare_answers_with_llm(expected: str, actual: str) -> dict[str, Any]:
    """Use an LLM judge to decide whether *actual* matches *expected*."""
    if "Error generating answer" in actual or "Rate limit reached" in actual:
        return {
            "match": False,
            "confidence": "high",
            "reason": "Answer generation failed",
        }

    system_prompt = (
        "You are an answer comparison expert. Compare the expected answer "
        "with the actual answer from a RAG system.\n\n"
        "Be LENIENT: if the actual answer contains the key information from "
        "the expected answer it is a match, even if it adds extra detail.\n\n"
        "Return JSON with: match (bool), confidence (high/medium/low), reason (string)."
    )
    user_prompt = (
        f"Expected: {expected}\n\nActual: {actual}\n\nCompare and return JSON."
    )

    try:
        load_dotenv()
        client = openai.OpenAI()

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=200,
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                return json.loads(content.strip())  # type: ignore[no-any-return]
            except Exception as exc:  # noqa: PERF203
                is_rate_limit = "429" in str(exc) or "rate limit" in str(exc).lower()
                if is_rate_limit and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return _simple_answer_comparison(expected, actual)

    except Exception:
        logger.debug("LLM answer comparison unavailable, using heuristic fallback")

    return _simple_answer_comparison(expected, actual)


# ---------------------------------------------------------------------------
# Retrieval quality grading
# ---------------------------------------------------------------------------


def _grade_retrieval(
    results: list[dict[str, Any]], expected_pages: list[int]
) -> dict[str, Any]:
    """Grade how well retrieval found the expected pages.

    Returns dict with keys: found, rank, grade, detail.
    """
    if not expected_pages:
        return {
            "found": False,
            "rank": None,
            "grade": "SPECIAL",
            "detail": "Information not expected to be in corpus",
        }

    for rank, result in enumerate(results, 1):
        if result["page"] in expected_pages:
            if rank <= 3:
                grade = "EXCELLENT"
            elif rank <= 5:
                grade = "GOOD"
            elif rank <= 10:
                grade = "FAIR"
            else:
                grade = "POOR"
            return {
                "found": True,
                "rank": rank,
                "grade": grade,
                "detail": f"Page {result['page']} at rank {rank}",
            }

    pages_str = ", ".join(map(str, expected_pages))
    return {
        "found": False,
        "rank": None,
        "grade": "FAILED",
        "detail": f"Expected pages {pages_str} not in top results",
    }


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "tq",
    EVAL_QUESTIONS,
    ids=[q.description for q in EVAL_QUESTIONS],
)
def test_search_quality(tq: TestQuestion):
    """Evaluate retrieval quality and answer accuracy for a single question."""
    result = search_codex(tq.question, k=20)

    # -- Retrieval --
    retrieval = _grade_retrieval(result["results"], tq.expected_pages)
    logger.info(
        "Retrieval [%s]: %s — %s",
        tq.description,
        retrieval["grade"],
        retrieval["detail"],
    )

    if tq.expected_pages:
        assert retrieval["found"], (
            f"Retrieval FAILED for '{tq.question}': {retrieval['detail']}"
        )

    # -- Answer accuracy --
    answer = result["answer"]
    assert answer, f"No answer generated for '{tq.question}'"

    comparison = compare_answers_with_llm(tq.expected_answer, answer)
    logger.info(
        "Answer   [%s]: match=%s confidence=%s — %s",
        tq.description,
        comparison["match"],
        comparison["confidence"],
        comparison.get("reason", ""),
    )
    logger.info("  Expected : %s", tq.expected_answer)
    logger.info("  Got      : %s", answer[:200])

    assert comparison["match"], (
        f"Answer mismatch for '{tq.question}'\n"
        f"  Expected: {tq.expected_answer}\n"
        f"  Got:      {answer[:300]}\n"
        f"  Reason:   {comparison.get('reason', '')}"
    )
