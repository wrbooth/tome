#!/usr/bin/env python3
"""
Codex Search Test Script

Tests search performance on specific historical questions with expected page numbers and answers.
"""

import subprocess
import re
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestQuestion:
    """Represents a test question with expected results."""
    question: str
    expected_page: int
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

def find_page_in_results(results: List[Dict], expected_page: int) -> Optional[Dict]:
    """Find a specific page in the search results."""
    for result in results:
        if result["page"] == expected_page:
            return result
    return None

def evaluate_search_performance(question: TestQuestion, results: List[Dict]) -> Dict:
    """Evaluate the performance of a search for a specific question."""
    expected_page = question.expected_page
    
    # Find the expected page in results
    page_result = find_page_in_results(results, expected_page)
    
    if page_result is None:
        return {
            "found": False,
            "rank": None,
            "score": None,
            "status": "FAILED - Expected page not found in top results"
        }
    
    rank = page_result["rank"]
    score = page_result["score"]
    
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
        "status": f"{status} - Page {expected_page} found at rank {rank}"
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
    
    for i, result in enumerate(test_results, 1):
        question = result["question"]
        evaluation = result["evaluation"]
        
        print(f"{i}. {question.question}")
        print(f"   Expected: Page {question.expected_page} - {question.expected_answer}")
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
            expected_page=15,
            expected_answer="1774",
            description="Military expedition timing"
        ),
        TestQuestion(
            question="What had the men involved in the Wills Creek Incident done that caused the Indians to come after them?",
            expected_page=55,
            expected_answer="They stole 15 horses from the Indians.",
            description="Wills Creek Incident cause"
        ),
        TestQuestion(
            question="In what years did the early settlers from the Isle of Guernsey arrive in Guernsey County?",
            expected_page=29,
            expected_answer="1806 and 1807",
            description="Guernsey settlers arrival years"
        ),
        TestQuestion(
            question="Did the Naftal family arrive in Cambridge in 1806?",
            expected_page=29,
            expected_answer="No.",
            description="Naftal family arrival verification"
        ),
        TestQuestion(
            question="What was the name of the company that built Cambridge's steel mill?",
            expected_page=50,
            expected_answer="The Cambridge Iron and Steel Company",
            description="Steel mill company identification"
        )
    ]
    
    print("Running Codex Search System Tests...")
    print(f"Testing {len(test_questions)} questions...")
    print()
    
    test_results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}/{len(test_questions)}: {question.question}")
        
        # Run the search
        search_result = run_search(question.question, k=20)
        
        if not search_result["success"]:
            print(f"  ERROR: {search_result['error']}")
            test_results.append({
                "question": question,
                "search_result": search_result,
                "evaluation": {
                    "found": False,
                    "rank": None,
                    "score": None,
                    "status": f"ERROR - {search_result['error']}"
                }
            })
            continue
        
        # Parse the results
        results = parse_search_results(search_result["output"])
        
        if not results:
            print(f"  ERROR: No results parsed from search output")
            test_results.append({
                "question": question,
                "search_result": search_result,
                "evaluation": {
                    "found": False,
                    "rank": None,
                    "score": None,
                    "status": "ERROR - No results parsed"
                }
            })
            continue
        
        # Evaluate performance
        evaluation = evaluate_search_performance(question, results)
        
        print(f"  Result: {evaluation['status']}")
        if evaluation["found"]:
            print(f"  Score: {evaluation['score']:.3f}")
        
        test_results.append({
            "question": question,
            "search_result": search_result,
            "evaluation": evaluation,
            "results": results
        })
        
        print()
    
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
