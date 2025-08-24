#!/usr/bin/env python3
"""
Codex Search Test Script

Tests search performance on specific historical questions with expected page numbers and answers.
"""

import subprocess
import re
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestQuestion:
    """Represents a test question with expected results."""
    question: str
    expected_pages: List[int]
    expected_answer: str
    description: str

def run_search(query: str, k: int = 20) -> Dict:
    """Run a search query and return the results."""
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "search.py", "--q", query, "--k", str(k)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Search failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        return {
            "success": True,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Search timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Exception running search: {e}"
        }

def parse_search_results(output: str) -> List[Dict]:
    """Parse search results from the output."""
    results = []
    
    # Look for the results section
    lines = output.split('\n')
    in_results = False
    
    for line in lines:
        if "Top " in line and "results:" in line:
            in_results = True
            continue
        
        if in_results and line.strip() == "":
            break
            
        if in_results and line.strip():
            # Parse result line: " 1  0.099  A Brief History of Guernsey County  p. 4  ..."
            match = re.match(r'\s*(\d+)\s+([\d.]+)\s+(.+?)\s+p\.\s+(\d+)\s+(.+)', line)
            if match:
                rank = int(match.group(1))
                score = float(match.group(2))
                title = match.group(3).strip()
                page = int(match.group(4))
                snippet = match.group(5).strip()
                
                results.append({
                    "rank": rank,
                    "score": score,
                    "title": title,
                    "page": page,
                    "snippet": snippet
                })
    
    return results

def find_page_in_results(results: List[Dict], expected_pages: List[int]) -> Optional[Dict]:
    """Find any of the expected pages in the search results."""
    for result in results:
        if result["page"] in expected_pages:
            return result
    return None

def evaluate_search_performance(question: TestQuestion, results: List[Dict]) -> Dict:
    """Evaluate the performance of a search for a specific question."""
    expected_pages = question.expected_pages
    
    # Special case: expected_pages=None or empty means the information is not in the book
    if expected_pages is None or len(expected_pages) == 0:
        # For questions where the answer is "No" or information is not available,
        # we expect that the search should not find relevant pages
        # This is a special case that needs manual evaluation
        return {
            "found": False,
            "rank": None,
            "score": None,
            "status": "SPECIAL CASE - Information not expected to be in book"
        }
    
    # Find any of the expected pages in results
    page_result = find_page_in_results(results, expected_pages)
    
    if page_result is None:
        pages_str = ", ".join(map(str, expected_pages))
        return {
            "found": False,
            "rank": None,
            "score": None,
            "status": f"FAILED - Expected pages {pages_str} not found in top results"
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
    
    return {
        "found": True,
        "rank": rank,
        "score": score,
        "status": f"{status} - Page {found_page} found at rank {rank}"
    }

def run_single_test(test_data: Tuple[int, TestQuestion]) -> Dict:
    """Run a single test and return the result."""
    test_num, question = test_data
    
    print(f"Test {test_num}: {question.question}")
    
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
                "status": f"ERROR - {search_result['error']}"
            }
        }
    
    # Parse the results
    results = parse_search_results(search_result["output"])
    
    if not results:
        print(f"  ERROR: No results parsed from search output")
        return {
            "test_num": test_num,
            "question": question,
            "search_result": search_result,
            "evaluation": {
                "found": False,
                "rank": None,
                "score": None,
                "status": "ERROR - No results parsed"
            }
        }
    
    # Evaluate performance
    evaluation = evaluate_search_performance(question, results)
    
    print(f"  Result: {evaluation['status']}")
    if evaluation["found"]:
        print(f"  Score: {evaluation['score']:.3f}")
    
    return {
        "test_num": test_num,
        "question": question,
        "search_result": search_result,
        "evaluation": evaluation,
        "results": results
    }

def print_test_results(test_results: List[Dict]):
    """Print formatted test results."""
    print("\n" + "="*80)
    print("SEARCH SYSTEM TEST RESULTS")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Summary statistics
    total_tests = len(test_results)
    found_count = sum(1 for r in test_results if r["evaluation"]["found"])
    excellent_count = sum(1 for r in test_results if "EXCELLENT" in r["evaluation"]["status"])
    good_count = sum(1 for r in test_results if "GOOD" in r["evaluation"]["status"])
    fair_count = sum(1 for r in test_results if "FAIR" in r["evaluation"]["status"])
    poor_count = sum(1 for r in test_results if "POOR" in r["evaluation"]["status"])
    failed_count = sum(1 for r in test_results if "FAILED" in r["evaluation"]["status"])
    
    print("SUMMARY:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Found Expected Page: {found_count}/{total_tests} ({found_count/total_tests*100:.1f}%)")
    print(f"  Excellent (Rank 1-3): {excellent_count}")
    print(f"  Good (Rank 4-5): {good_count}")
    print(f"  Fair (Rank 6-10): {fair_count}")
    print(f"  Poor (Rank >10): {poor_count}")
    print(f"  Failed (Not Found): {failed_count}")
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
        print(f"   Result: {evaluation['status']}")
        
        if evaluation["found"]:
            print(f"   Score: {evaluation['score']:.3f}")
        
        print()

def main():
    """Run the search system tests."""
    
    # Define test questions
    test_questions = [
        TestQuestion(
            question="When did the McDonald military expedition cross Guernsey County?",
            expected_pages=[15],
            expected_answer="1774",
            description="Military expedition timing"
        ),
        TestQuestion(
            question="What had the men involved in the Wills Creek Incident done that caused the Indians to come after them?",
            expected_pages=[55],
            expected_answer="They stole 15 horses from the Indians.",
            description="Wills Creek Incident cause"
        ),
        TestQuestion(
            question="In what years did the early settlers from the Isle of Guernsey arrive in Guernsey County?",
            expected_pages=[29],
            expected_answer="1806 and 1807",
            description="Guernsey settlers arrival years"
        ),
        TestQuestion(
            question="Did the Naftal family arrive in Cambridge in 1806?",
            expected_pages=[29],
            expected_answer="No.",
            description="Naftal family arrival verification"
        ),
        TestQuestion(
            question="What was the name of the company that built Cambridge's steel mill?",
            expected_pages=[50],
            expected_answer="The Cambridge Iron and Steel Company",
            description="Steel mill company identification"
        ),
        TestQuestion(
            question="When did Cambridge get its second major railroad going north and south?",
            expected_pages=[43],
            expected_answer="1873",
            description="Cambridge railroad expansion"
        ),
        TestQuestion(
            question="What was the old name for Maysville, Kentucky, in the 1700s?",
            expected_pages=[19],
            expected_answer="Limestone",
            description="Maysville historical name"
        ),
        TestQuestion(
            question="When did Morgan's raid reach Cumberland?",
            expected_pages=[45],
            expected_answer="July 23, 1863",
            description="Morgan's raid timing"
        ),

        TestQuestion(
            question="What can you tell me about an army hospital built near Cambridge?",
            expected_pages=[51],
            expected_answer="the Fletcher General Hospital story",
            description="Army hospital information"
        ),
        TestQuestion(
            question="When and where did glass manufacturing start in Guernsey County?",
            expected_pages=[50],
            expected_answer="1884, in Quaker City",
            description="Glass manufacturing history"
        ),
        TestQuestion(
            question="When did Cambridge's steel mill go out of business?",
            expected_pages=[50],
            expected_answer="in the 1940s",
            description="Steel mill closure"
        ),
        TestQuestion(
            question="Who were the main historians of Guernsey County?",
            expected_pages=[9],
            expected_answer="main historians of Guernsey County",
            description="County historians identification"
        ),
        TestQuestion(
            question="When did Congress authorize building the National Road?",
            expected_pages=[38],
            expected_answer="1802",
            description="National Road authorization"
        ),
        TestQuestion(
            question="Did any of the founders of Cambridge participate in the Revolutionary War?",
            expected_pages=[27],
            expected_answer="Jacob Gomber did",
            description="Cambridge founders Revolutionary War participation"
        ),
        TestQuestion(
            question="Who was the first sitting president to pass through Cambridge?",
            expected_pages=[58],
            expected_answer="James Monroe",
            description="First president to visit Cambridge"
        ),
        TestQuestion(
            question="Who built the Colonial Theater, and when?",
            expected_pages=[92],
            expected_answer="not clear who built it, but it came to be around 1901",
            description="Colonial Theater construction"
        ),
        TestQuestion(
            question="Was John Glenn ever in combat?",
            expected_pages=[51],
            expected_answer="Yes.",
            description="John Glenn combat experience"
        ),
        TestQuestion(
            question="What sort of things did Morgan's Raiders steal?",
            expected_pages=[62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
            expected_answer="lots of things, but especially horses",
            description="Morgan's Raiders stolen items"
        ),
        TestQuestion(
            question="Whose farm did the army take over to build Fletcher General Hospital?",
            expected_pages=[],
            expected_answer="not mentioned in the book; other books may mention it was the Oldham farm",
            description="Fletcher General Hospital farm ownership"
        ),
        TestQuestion(
            question="Did Morgan's Raid pass through Byesville?",
            expected_pages=[],
            expected_answer="not on route mentioned in the book; No.",
            description="Morgan's Raid route through Byesville"
        )
    ]
    
    print("Running Codex Search System Tests...")
    print(f"Testing {len(test_questions)} questions in parallel...")
    print()
    
    # Prepare test data for parallel processing
    test_data = [(i, question) for i, question in enumerate(test_questions, 1)]
    
    # Determine number of workers (use CPU count, but cap at 8 to avoid overwhelming the system)
    max_workers = min(mp.cpu_count(), 8)
    print(f"Using {max_workers} parallel workers...")
    
    test_results = []
    
    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tests
        future_to_test = {executor.submit(run_single_test, test_data[i]): i for i in range(len(test_data))}
        
        # Collect results as they complete
        for future in as_completed(future_to_test):
            try:
                result = future.result()
                test_results.append(result)
            except Exception as exc:
                test_num = future_to_test[future] + 1
                print(f"Test {test_num} generated an exception: {exc}")
                test_results.append({
                    "test_num": test_num,
                    "question": test_questions[future_to_test[future]],
                    "search_result": {"success": False, "error": str(exc)},
                    "evaluation": {
                        "found": False,
                        "rank": None,
                        "score": None,
                        "status": f"ERROR - Exception: {exc}"
                    }
                })
    
    # Print comprehensive results
    print_test_results(test_results)
    
    # Return exit code based on performance
    failed_count = sum(1 for r in test_results if "FAILED" in r["evaluation"]["status"] or "ERROR" in r["evaluation"]["status"])
    if failed_count > 0:
        print(f"❌ {failed_count} tests failed")
        sys.exit(1)
    else:
        print("✅ All tests completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
