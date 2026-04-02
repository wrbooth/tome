#!/usr/bin/env python3
"""
Codex Search Script

Implements hybrid search via Meilisearch with cross-encoder re-ranking.
"""

import os
import sys
import click
from psycopg2.extras import RealDictCursor
import json
from typing import List, Dict, Any, Tuple, Optional

from config import get_db_connection, get_meili_client, get_openai_client

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze a query in a single LLM call using structured output.
    Returns classification, entities, person name, and search expansions.
    """
    default_result = {
        "query_type": "general",
        "person": None,
        "entities": {
            "persons": [], "places": [], "events": [], "dates": [],
            "families": [], "companies": [], "industries": [], "settlement_terms": []
        },
        "expansions": [query]
    }

    try:
        system_prompt = """You are a query analysis engine for a historical document search system.
Given a user query, produce a JSON object with the following fields:

1. **query_type**: Classify the query as one of:
   - "who": Questions asking about identity or role
   - "when": Questions about timing, dates, or years
   - "where": Questions about locations or places
   - "person": Questions about specific individuals, their actions, or personal history
   - "company": Questions about businesses, organizations, or institutions
   - "factoid": Questions about specific facts, events, or details
   - "general": General information requests that don't fit other categories

2. **person**: The name of the person being asked about. Use empty string if none.
   Examples: "Who was the first president to..." → "first president"; "Was John Smith ever in combat?" → "John Smith"

3. **entities**: Extract entities into these categories:
   - persons: Individual people mentioned
   - places: Locations, cities, counties, states, countries
   - events: Historical events or incidents
   - dates: Years, decades, centuries (e.g., "1806", "1800s", "19th century")
   - families: Family names or groups
   - companies: Businesses, organizations, corporations
   - industries: Industry types or sectors
   - settlement_terms: Terms related to settlement, immigration, arrival
   For temporal queries, include relevant historical periods. For "who" questions about roles, include relevant historical figures.

4. **expansions**: 5-10 additional search terms to help find relevant passages. Include:
   - Synonyms and related concepts
   - Specific names, dates, or places mentioned or implied
   - Broader and narrower terms
   - Historical context terms
   - Alternative phrasings
   - Key entity combinations (e.g., "[location] settlers")
   For "who" questions about roles, focus on role/position terms. Be specific to context."""

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": ["who", "when", "where", "person", "company", "factoid", "general"]
                            },
                            "person": {"type": "string"},
                            "entities": {
                                "type": "object",
                                "properties": {
                                    "persons": {"type": "array", "items": {"type": "string"}},
                                    "places": {"type": "array", "items": {"type": "string"}},
                                    "events": {"type": "array", "items": {"type": "string"}},
                                    "dates": {"type": "array", "items": {"type": "string"}},
                                    "families": {"type": "array", "items": {"type": "string"}},
                                    "companies": {"type": "array", "items": {"type": "string"}},
                                    "industries": {"type": "array", "items": {"type": "string"}},
                                    "settlement_terms": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["persons", "places", "events", "dates", "families", "companies", "industries", "settlement_terms"],
                                "additionalProperties": False
                            },
                            "expansions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["query_type", "person", "entities", "expansions"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(response.choices[0].message.content)

        # Normalize person field
        if not result["person"] or result["person"].lower() == "none":
            result["person"] = None

        # Ensure original query is first in expansions
        expansions = result.get("expansions", [])
        if query not in expansions:
            expansions.insert(0, query)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique.append(exp)
        result["expansions"] = unique[:15]

        return result

    except Exception as e:
        print(f"Error in query analysis: {e}, using defaults")
        return default_result

def hybrid_search(query: str, k: int = 200, document_id: Optional[str] = None,
                  semantic_ratio: float = 0.75) -> List[Tuple[str, float]]:
    """Run hybrid (keyword + semantic) search via Meilisearch."""
    try:
        client = get_meili_client()
        index = client.index("passages")

        search_params = {
            "hybrid": {
                "semanticRatio": semantic_ratio,
                "embedder": "default"
            },
            "limit": k,
            "attributesToRetrieve": ["id"],
            "showRankingScore": True
        }

        if document_id:
            search_params["filter"] = f"document_id = '{document_id}'"

        response = index.search(query, search_params)

        return [
            (hit["id"], hit.get("_rankingScore", 0.0))
            for hit in response["hits"]
        ]

    except Exception as e:
        print(f"Warning: Hybrid search failed: {e}")
        return []

def _get_reranker():
    """Lazy-load the cross-encoder re-ranker model."""
    if not hasattr(_get_reranker, "_model"):
        from sentence_transformers import CrossEncoder
        _get_reranker._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _get_reranker._model

def rerank_candidates(candidates: List[Tuple[str, float]], query: str, conn, k: int = 20) -> List[Tuple[str, float]]:
    """Re-rank candidates using a cross-encoder model."""
    if not candidates:
        return candidates

    # Fetch passage texts for all candidates in one query
    passage_ids = [pid for pid, _ in candidates]
    placeholders = ','.join(['%s'] * len(passage_ids))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT id, text FROM passages WHERE id IN ({placeholders})
        """, passage_ids)
        id_to_text = {row['id']: row['text'] for row in cur.fetchall()}

    # Build query-passage pairs for the cross-encoder
    pairs = []
    valid_ids = []
    for pid, _ in candidates:
        text = id_to_text.get(pid)
        if text:
            pairs.append((query, text))
            valid_ids.append(pid)

    if not pairs:
        return candidates

    # Score all pairs in one batch
    reranker = _get_reranker()
    scores = reranker.predict(pairs)

    # Sort by cross-encoder score descending
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [(pid, float(score)) for pid, score in ranked[:k]]

def get_passage_details(conn, passage_ids: List[str]) -> List[Dict[str, Any]]:
    """Get detailed passage information."""
    if not passage_ids:
        return []
    
    placeholders = ','.join(['%s'] * len(passage_ids))
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT p.id, p.text, p.page, d.title, p.headings_path
            FROM passages p
            JOIN documents d ON p.document_id = d.id
            WHERE p.id IN ({placeholders})
        """, passage_ids)
        
        results = cur.fetchall()
        
        # Preserve the order of passage_ids
        id_to_result = {result['id']: result for result in results}
        ordered_results = []
        for passage_id in passage_ids:
            if passage_id in id_to_result:
                ordered_results.append(id_to_result[passage_id])
        
        return ordered_results

def format_results(results: List[Dict[str, Any]], scores: Dict[str, float]) -> str:
    """Format search results for display."""
    output = []
    
    # Check if we have multiple documents
    document_titles = set(result.get('title', 'Unknown') for result in results)
    multi_document = len(document_titles) > 1
    
    for i, result in enumerate(results, 1):
        passage_id = result['id']
        score = scores.get(passage_id, 0.0)
        title = result['title']
        page = result['page']
        
        # Truncate text for snippet
        text = result['text']
        snippet = text[:200] + "..." if len(text) > 200 else text
        
        # Show document title more prominently in multi-document scenarios
        if multi_document:
            output.append(f"{i:2d}  {score:.3f}  [{title}]  p. {page}  \"{snippet}\"")
        else:
            output.append(f"{i:2d}  {score:.3f}  {title}  p. {page}  \"{snippet}\"")
    
    return '\n'.join(output)

def generate_answer(query: str, query_type: str, results: List[Dict[str, Any]]) -> str:
    """Generate an LLM-based answer from search results."""
    if not results:
        return "I couldn't find any relevant information to answer your question."
    
    try:
        # Import the answer generator
        from answer_generator import answer_query
        
        # Generate answer using LLM
        answer = answer_query(query, results)
        return answer
        
    except ImportError:
        # Fallback to simple answer if LLM module not available
        top_result = results[0]
        title = top_result.get('title', 'Unknown Document')
        return f"Top result: {top_result['text'][:300]}... (Source: {title}, p. {top_result['page']})"
    except Exception as e:
        # Fallback to simple answer if LLM fails
        top_result = results[0]
        title = top_result.get('title', 'Unknown Document')
        return f"Top result: {top_result['text'][:300]}... (Source: {title}, p. {top_result['page']})\n\nNote: LLM answer generation failed: {str(e)}"

@click.command()
@click.option('--q', 'query', required=True, help='Search query')
@click.option('--k', default=20, type=int, help='Number of results to return')
@click.option('--doc', 'document_id', help='Filter by document ID')
def main(query: str, k: int, document_id: Optional[str]):
    """Search passages using hybrid retrieval with RRF fusion."""
    
    print(f"Searching for: {query}")

    # Analyze query (single LLM call for classification, entities, and expansions)
    query_analysis = analyze_query(query)
    query_type = query_analysis["query_type"]
    print(f"Query type: {query_type}")

    # Hybrid search (keyword + semantic in one Meilisearch call)
    print("Running hybrid search...")
    candidates = hybrid_search(query, k=200, document_id=document_id)
    print(f"Found {len(candidates)} candidates")

    if not candidates:
        print("No candidates found")
        return

    # Re-rank with cross-encoder
    conn = get_db_connection()
    print("Re-ranking with cross-encoder...")
    reranked_candidates = rerank_candidates(candidates, query, conn, k=k)
    
    # Get passage details
    passage_ids = [pid for pid, _ in reranked_candidates]
    passage_details = get_passage_details(conn, passage_ids)

    # Create score mapping
    score_map = {pid: score for pid, score in reranked_candidates}
    
    # Generate answer
    answer = generate_answer(query, query_type, passage_details)
    print(f"\nAnswer:")
    print("=" * 80)
    print(answer)
    
    # Format and display results
    print(f"\nTop {len(passage_details)} results:")
    print("=" * 80)
    print(format_results(passage_details, score_map))
    
    conn.close()

if __name__ == "__main__":
    main()








