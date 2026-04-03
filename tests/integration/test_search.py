#!/usr/bin/env python3
"""
Tome Search Test Script

Tests search performance on specific historical questions
with expected page numbers and answers.
"""

import multiprocessing as mp
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

import openai
from dotenv import load_dotenv


@dataclass
class TestQuestion:
    """Represents a test question with expected results."""

    question: str
    expected_pages: list[int]
    expected_answer: str
    description: str


def run_search(query: str, k: int = 20) -> dict:
    """Run a search query and return the results."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "tome.search.search.py", "--q", query, "--k", str(k)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Search failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        return {"success": True, "output": result.stdout, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Search timed out after 120 seconds"}
    except Exception as e:
        return {"success": False, "error": f"Exception running search: {e}"}


def extract_llm_answer(output: str) -> str | None:
    """Extract the LLM answer from search output."""
    lines = output.split("\n")
    in_answer = False
    answer_lines = []

    for _, line in enumerate(lines):
        # Look for the Answer section
        if "Answer:" in line:
            in_answer = True
            # Extract the answer part after "Answer:"
            answer_part = line.split("Answer:", 1)[1].strip()
            if answer_part:
                answer_lines.append(answer_part)
            continue

        # If we're in answer mode, collect lines until we hit the results section
        if in_answer:
            # Stop when we hit the results section
            if "Top " in line and "results:" in line:
                break

            # Stop when we hit a separator line (but not the first one after Answer:)
            if line.strip().startswith("=" * 20):
                # Skip the first separator line after Answer:
                if len(answer_lines) == 0:
                    continue
                break

            if line.strip():
                answer_lines.append(line.strip())

    if answer_lines:
        answer = "\n".join(answer_lines)
        # Clean up any remaining separator lines or formatting artifacts
        answer = re.sub(r"=+\s*", "", answer)
        answer = re.sub(r"\n\s*\n+", "\n", answer)
        return answer.strip()
    return None


def parse_search_results(output: str) -> list[dict]:
    """Parse search results from the output."""
    results = []

    # Look for the results section
    lines = output.split("\n")
    in_results = False

    for line in lines:
        if "Top " in line and "results:" in line:
            in_results = True
            continue

        if in_results and line.strip() == "":
            break

        if in_results and line.strip():
            # Parse result line:
            # " 1  0.099  A Brief History...  p. 4  ..."
            match = re.match(
                r"\s*(\d+)\s+(-?[\d.]+)\s+(.+?)\s+p\.\s+(\d+)\s+(.+)", line
            )
            if match:
                rank = int(match.group(1))
                score = float(match.group(2))
                title = match.group(3).strip()
                page = int(match.group(4))
                snippet = match.group(5).strip()

                results.append(
                    {
                        "rank": rank,
                        "score": score,
                        "title": title,
                        "page": page,
                        "snippet": snippet,
                    }
                )

    return results


def find_page_in_results(results: list[dict], expected_pages: list[int]) -> dict | None:
    """Find any of the expected pages in the search results."""
    for result in results:
        if result["page"] in expected_pages:
            return result
    return None


def compare_answers_with_llm(expected_answer: str, actual_answer: str) -> dict:  # noqa: C901
    """Use LLM to compare expected and actual answers."""
    import json
    import time

    # Check for error messages in actual answer
    if (
        "Error generating answer" in actual_answer
        or "Rate limit reached" in actual_answer
    ):
        return {
            "match": False,
            "confidence": "high",
            "reason": "LLM answer generation failed due to rate limiting or error",
        }

    # Simple fallback comparison for basic cases
    def simple_fallback_comparison(expected: str, actual: str) -> dict:
        """Simple text-based comparison as fallback."""
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Check for exact matches or key phrases
        if expected_lower in actual_lower:
            return {
                "match": True,
                "confidence": "medium",
                "reason": "Key information found in actual answer",
            }

        # Check for "not mentioned" or "not in book" cases
        if (
            "not mentioned" in expected_lower or "not in book" in expected_lower
        ) and any(
            phrase in actual_lower
            for phrase in [
                "not provide",
                "does not",
                "no information",
                "not mention",
            ]
        ):
            return {
                "match": True,
                "confidence": "medium",
                "reason": "Both indicate information not available",
            }

        # Check for "No" answers
        if expected_lower.strip() == "no." and (
            "no" in actual_lower
            or "did not" in actual_lower
            or "was not" in actual_lower
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

    try:
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            return {
                "match": False,
                "confidence": "low",
                "reason": "OpenAI API key not available",
            }

        system_prompt = (
            "You are an answer comparison expert. "
            "Compare the expected answer with the actual "
            "answer provided by an AI system.\n\n"
            "Evaluate whether the actual answer:\n"
            "1. Contains the key information from the "
            "expected answer\n"
            "2. Is factually accurate according to the "
            "expected answer\n"
            "3. Addresses the same question/point\n\n"
            "For questions where the expected answer "
            'indicates information is "not mentioned" or '
            '"not in the book", the actual answer should '
            "also indicate this.\n\n"
            "IMPORTANT: Be VERY lenient with detailed "
            "answers. If the actual answer contains the "
            "key information from the expected answer, it "
            "should be considered a match, even if it "
            "provides additional context, elaboration, or "
            "details. The goal is to check if the core "
            "information is present and accurate, not to "
            "require exact brevity.\n\n"
            "Examples of what should be considered "
            "matches:\n"
            '- Expected: "1774" vs Actual: "The '
            "expedition crossed in 1774 during the summer "
            'months" \u2192 MATCH\n'
            '- Expected: "Yes." vs Actual: "Yes, John '
            "Glenn was in combat during World War II and "
            'Korean War" \u2192 MATCH\n'
            '- Expected: "James Monroe" vs Actual: '
            '"President James Monroe was the first '
            'sitting president to visit" \u2192 MATCH\n'
            '- Expected: "No." vs Actual: "No, the '
            'Naftal family did not arrive in 1806" '
            "\u2192 MATCH\n\n"
            "Return a JSON response with:\n"
            '- "match": true/false (whether the answers '
            "are essentially equivalent)\n"
            '- "confidence": "high"/"medium"/"low" '
            "(confidence in the comparison)\n"
            '- "reason": brief explanation of the '
            "comparison result\n\n"
            "Focus on factual accuracy and completeness, "
            "not exact wording."
        )

        user_prompt = f"""Expected Answer: {expected_answer}

Actual Answer: {actual_answer}

Compare these answers and provide your evaluation."""

        client = openai.OpenAI()

        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
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

                try:
                    return json.loads(response.choices[0].message.content.strip())
                except json.JSONDecodeError:
                    return {
                        "match": False,
                        "confidence": "low",
                        "reason": "Failed to parse LLM comparison response",
                    }

            except Exception as e:  # noqa: PERF203
                error_str = str(e)
                if "rate limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        print(f"  Rate limit hit, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    return {
                        "match": False,
                        "confidence": "low",
                        "reason": "Rate limit exceeded after retries",
                    }
                return {
                    "match": False,
                    "confidence": "low",
                    "reason": f"LLM comparison failed: {error_str}",
                }

        # If all retries failed, use fallback comparison
        return simple_fallback_comparison(expected_answer, actual_answer)

    except Exception:
        # Use fallback comparison if LLM completely fails
        return simple_fallback_comparison(expected_answer, actual_answer)


def evaluate_search_performance(
    question: TestQuestion, results: list[dict], llm_answer: str | None = None
) -> dict:
    """Evaluate the performance of a search for a specific question."""
    expected_pages = question.expected_pages

    # Special case: expected_pages=None or empty means
    # the information is not in the book
    if expected_pages is None or len(expected_pages) == 0:
        # For questions where the answer is "No" or information is not available,
        # we expect that the search should not find relevant pages
        # This is a special case that needs manual evaluation

        # Add LLM answer comparison if available
        answer_evaluation = None
        answer_status = "NO ANSWER"
        if llm_answer:
            answer_evaluation = compare_answers_with_llm(
                question.expected_answer, llm_answer
            )
            if answer_evaluation["match"]:
                answer_status = "CORRECT ANSWER"
            else:
                answer_status = "INCORRECT ANSWER"

        return {
            "found": False,
            "rank": None,
            "score": None,
            "chunk_status": "SPECIAL CASE - Information not expected to be in book",
            "answer_status": answer_status,
            "answer_evaluation": answer_evaluation,
        }

    # Find any of the expected pages in results
    page_result = find_page_in_results(results, expected_pages)

    if page_result is None:
        pages_str = ", ".join(map(str, expected_pages))
        return {
            "found": False,
            "rank": None,
            "score": None,
            "chunk_status": (
                f"FAILED - Expected pages {pages_str} not found in top results"
            ),
            "answer_status": "NO ANSWER",
        }

    rank = page_result["rank"]
    score = page_result["score"]
    found_page = page_result["page"]

    # Determine status based on rank
    if rank <= 3:
        status = "EXCELLENT"
    elif rank <= 5:
        status = "GOOD"
    elif rank <= 10:
        status = "FAIR"
    else:
        status = "POOR"

    # Add LLM answer comparison if available
    answer_evaluation = None
    answer_status = "NO ANSWER"
    if llm_answer:
        answer_evaluation = compare_answers_with_llm(
            question.expected_answer, llm_answer
        )
        if answer_evaluation["match"]:
            answer_status = "CORRECT ANSWER"
        else:
            answer_status = "INCORRECT ANSWER"

    return {
        "found": True,
        "rank": rank,
        "score": score,
        "chunk_status": f"{status} - Page {found_page} found at rank {rank}",
        "answer_status": answer_status,
        "answer_evaluation": answer_evaluation,
    }


def run_single_test(test_data: tuple[int, TestQuestion], delay: float = 1.0) -> dict:
    """Run a single test and return the result."""
    test_num, question = test_data

    print(f"Test {test_num}: {question.question}")

    # Add a delay to help with rate limiting
    import time

    time.sleep(delay)

    # Run the search
    search_result = run_search(question.question, k=20)

    if not search_result["success"]:
        print(f"  ERROR: {search_result['error']}")
        return {
            "test_num": test_num,
            "question": question,
            "search_result": search_result,
            "evaluation": {
                "found": False,
                "rank": None,
                "score": None,
                "chunk_status": f"ERROR - {search_result['error']}",
                "answer_status": "NO ANSWER",
            },
        }

    # Parse the results
    results = parse_search_results(search_result["output"])

    if not results:
        print("  ERROR: No results parsed from search output")
        return {
            "test_num": test_num,
            "question": question,
            "search_result": search_result,
            "evaluation": {
                "found": False,
                "rank": None,
                "score": None,
                "chunk_status": "ERROR - No results parsed",
                "answer_status": "NO ANSWER",
            },
        }

    # Extract LLM answer
    llm_answer = extract_llm_answer(search_result["output"])

    # Evaluate performance
    evaluation = evaluate_search_performance(question, results, llm_answer)

    print(f"  Chunk Result: {evaluation['chunk_status']}")
    if evaluation["found"]:
        print(f"  Score: {evaluation['score']:.3f}")
    print(f"  Answer Result: {evaluation['answer_status']}")

    if llm_answer:
        print(f"  LLM Answer: {llm_answer[:100]}...")

    return {
        "test_num": test_num,
        "question": question,
        "search_result": search_result,
        "evaluation": evaluation,
        "results": results,
        "llm_answer": llm_answer,
    }


def print_test_results(test_results: list[dict]):
    """Print formatted test results."""
    print("\n" + "=" * 80)
    print("SEARCH SYSTEM TEST RESULTS")
    print("=" * 80)
    print(f"Test Date: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Summary statistics
    total_tests = len(test_results)
    found_count = sum(1 for r in test_results if r["evaluation"]["found"])
    excellent_count = sum(
        1 for r in test_results if "EXCELLENT" in r["evaluation"]["chunk_status"]
    )
    good_count = sum(
        1 for r in test_results if "GOOD" in r["evaluation"]["chunk_status"]
    )
    fair_count = sum(
        1 for r in test_results if "FAIR" in r["evaluation"]["chunk_status"]
    )
    poor_count = sum(
        1 for r in test_results if "POOR" in r["evaluation"]["chunk_status"]
    )
    failed_count = sum(
        1 for r in test_results if "FAILED" in r["evaluation"]["chunk_status"]
    )

    # Answer accuracy statistics
    correct_answers = sum(
        1
        for r in test_results
        if r["evaluation"].get("answer_evaluation")
        and r["evaluation"]["answer_evaluation"].get("match", False)
    )
    total_with_answers = sum(1 for r in test_results if r.get("llm_answer"))

    print("SUMMARY:")
    print(f"  Total Tests: {total_tests}")
    print("  Chunk Retrieval Performance:")
    print(
        f"    Found Expected Page: {found_count}/{total_tests}"
        f" ({found_count / total_tests * 100:.1f}%)"
    )
    print(f"    Excellent (Rank 1-3): {excellent_count}")
    print(f"    Good (Rank 4-5): {good_count}")
    print(f"    Fair (Rank 6-10): {fair_count}")
    print(f"    Poor (Rank >10): {poor_count}")
    print(f"    Failed (Not Found): {failed_count}")
    print("  Answer Generation Performance:")
    if total_with_answers > 0:
        print(
            f"    Correct Answers: {correct_answers}"
            f"/{total_with_answers}"
            f" ({correct_answers / total_with_answers * 100:.1f}%)"
        )
    else:
        print("    No LLM answers generated")
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)

    # Sort results by test number to maintain order
    sorted_results = sorted(test_results, key=lambda x: x["test_num"])

    for result in sorted_results:
        question = result["question"]
        evaluation = result["evaluation"]

        print(f"{result['test_num']}. {question.question}")
        if question.expected_pages:
            pages_str = ", ".join(map(str, question.expected_pages))
            print(f"   Expected: Pages {pages_str} - {question.expected_answer}")
        else:
            print(f"   Expected: Not in book - {question.expected_answer}")
        print(f"   Chunk Result: {evaluation['chunk_status']}")

        if evaluation["found"]:
            print(f"   Score: {evaluation['score']:.3f}")

        print(f"   Answer Result: {evaluation['answer_status']}")

        # Show answer comparison if available
        if result.get("llm_answer"):
            print("   LLM Answer:")
            print(f"     {result['llm_answer']}")
            if evaluation.get("answer_evaluation"):
                ae = evaluation["answer_evaluation"]
                print(
                    f"   Answer Match: {ae.get('match', False)}"
                    f" (Confidence: {ae.get('confidence', 'unknown')})"
                )
                print(f"   Reason: {ae.get('reason', 'No reason provided')}")
        else:
            print("   LLM Answer: No answer generated")

        print()


def main():
    """Run the search system tests."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Tome search system tests")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run tests serially instead of in parallel (helps with rate limiting)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between tests in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    # Define test questions
    test_questions = [
        TestQuestion(
            question="When did the McDonald military expedition cross Guernsey County?",
            expected_pages=[15],
            expected_answer="1774",
            description="Military expedition timing",
        ),
        TestQuestion(
            question=(
                "What had the men involved in the Wills"
                " Creek Incident done that caused the"
                " Indians to come after them?"
            ),
            expected_pages=[55],
            expected_answer="They stole 15 horses from the Indians.",
            description="Wills Creek Incident cause",
        ),
        TestQuestion(
            question=(
                "In what years did the early settlers"
                " from the Isle of Guernsey arrive in"
                " Guernsey County?"
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
            question=(
                "What was the name of the company that built Cambridge's steel mill?"
            ),
            expected_pages=[50],
            expected_answer="The Cambridge Iron and Steel Company",
            description="Steel mill company identification",
        ),
        TestQuestion(
            question=(
                "When did Cambridge get its second"
                " major railroad going north and south?"
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
            question=(
                "What can you tell me about an army hospital built near Cambridge?"
            ),
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
                "Did any of the founders of Cambridge"
                " participate in the Revolutionary War?"
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
            question=(
                "Whose farm did the army take over to build Fletcher General Hospital?"
            ),
            expected_pages=[],
            expected_answer=(
                "not mentioned in the book; other books"
                " may mention it was the Oldham farm"
            ),
            description="Fletcher General Hospital farm ownership",
        ),
        TestQuestion(
            question="Did Morgan's Raid pass through Byesville?",
            expected_pages=[],
            expected_answer="not on route mentioned in the book; No.",
            description="Morgan's Raid route through Byesville",
        ),
    ]

    print("Running Tome Search System Tests...")

    if args.serial:
        print(f"Testing {len(test_questions)} questions serially...")
        print()

        test_results = []

        # Run tests serially
        for i, question in enumerate(test_questions, 1):
            test_data = (i, question)
            try:
                result = run_single_test(test_data, args.delay)
                test_results.append(result)
            except Exception as exc:
                print(f"Test {i} generated an exception: {exc}")
                test_results.append(
                    {
                        "test_num": i,
                        "question": question,
                        "search_result": {"success": False, "error": str(exc)},
                        "evaluation": {
                            "found": False,
                            "rank": None,
                            "score": None,
                            "chunk_status": f"ERROR - Exception: {exc}",
                            "answer_status": "NO ANSWER",
                        },
                    }
                )
    else:
        print(f"Testing {len(test_questions)} questions in parallel...")
        print()

        # Prepare test data for parallel processing
        test_data = [(i, question) for i, question in enumerate(test_questions, 1)]

        # Determine number of workers
        max_workers = min(mp.cpu_count(), args.workers)
        print(f"Using {max_workers} parallel workers...")

        test_results = []

        # Run tests in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(run_single_test, test_data[i]): i
                for i in range(len(test_data))
            }

            # Collect results as they complete
            for future in as_completed(future_to_test):
                try:
                    result = future.result()
                    test_results.append(result)
                except Exception as exc:  # noqa: PERF203
                    test_num = future_to_test[future] + 1
                    print(f"Test {test_num} generated an exception: {exc}")
                    test_results.append(
                        {
                            "test_num": test_num,
                            "question": test_questions[future_to_test[future]],
                            "search_result": {"success": False, "error": str(exc)},
                            "evaluation": {
                                "found": False,
                                "rank": None,
                                "score": None,
                                "chunk_status": f"ERROR - Exception: {exc}",
                                "answer_status": "NO ANSWER",
                            },
                        }
                    )

    # Print comprehensive results
    print_test_results(test_results)

    # Return exit code based on performance
    failed_count = sum(
        1
        for r in test_results
        if "FAILED" in r["evaluation"]["chunk_status"]
        or "ERROR" in r["evaluation"]["chunk_status"]
    )
    if failed_count > 0:
        print(f"❌ {failed_count} tests failed")
        sys.exit(1)
    else:
        print("✅ All tests completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
