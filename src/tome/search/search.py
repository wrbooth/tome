#!/usr/bin/env python3
"""
Tome Search Script

Implements hybrid search via Meilisearch with cross-encoder re-ranking.
"""

import json
import logging
import uuid
from typing import Any

import click
from psycopg2.extras import RealDictCursor

from tome.config import (
    QUERY_ANALYSIS_MODEL,
    RERANKER_MODEL,
    configure_logging,
    db_connection,
    get_meili_client,
    get_openai_client,
)

logger = logging.getLogger(__name__)


def analyze_query(query: str) -> dict[str, Any]:
    """
    Analyze a query in a single LLM call using structured output.
    Returns classification, entities, person name, and search expansions.
    """
    default_result = {
        "query_type": "general",
        "person": None,
        "entities": {
            "persons": [],
            "places": [],
            "events": [],
            "dates": [],
            "families": [],
            "companies": [],
            "industries": [],
            "settlement_terms": [],
        },
        "expansions": [query],
    }

    try:
        system_prompt = (
            "You are a query analysis engine for a "
            "historical document search system.\n"
            "Given a user query, produce a JSON object "
            "with the following fields:\n\n"
            "1. **query_type**: Classify the query as one of:\n"
            '   - "who": Questions asking about identity or role\n'
            '   - "when": Questions about timing, dates, or years\n'
            '   - "where": Questions about locations or places\n'
            '   - "person": Questions about specific individuals,'
            " their actions, or personal history\n"
            '   - "company": Questions about businesses,'
            " organizations, or institutions\n"
            '   - "factoid": Questions about specific facts,'
            " events, or details\n"
            '   - "general": General information requests that'
            " don't fit other categories\n\n"
            "2. **person**: The name of the person being asked "
            "about. Use empty string if none.\n"
            '   Examples: "Who was the first president to..."'
            ' -> "first president"; "Was John Smith ever in'
            ' combat?" -> "John Smith"\n\n'
            "3. **entities**: Extract entities into these "
            "categories:\n"
            "   - persons: Individual people mentioned\n"
            "   - places: Locations, cities, counties, states,"
            " countries\n"
            "   - events: Historical events or incidents\n"
            "   - dates: Years, decades, centuries "
            '(e.g., "1806", "1800s", "19th century")\n'
            "   - families: Family names or groups\n"
            "   - companies: Businesses, organizations,"
            " corporations\n"
            "   - industries: Industry types or sectors\n"
            "   - settlement_terms: Terms related to settlement,"
            " immigration, arrival\n"
            "   For temporal queries, include relevant historical"
            ' periods. For "who" questions about roles,'
            " include relevant historical figures.\n\n"
            "4. **expansions**: 5-10 additional search terms to "
            "help find relevant passages. Include:\n"
            "   - Synonyms and related concepts\n"
            "   - Specific names, dates, or places mentioned"
            " or implied\n"
            "   - Broader and narrower terms\n"
            "   - Historical context terms\n"
            "   - Alternative phrasings\n"
            '   - Key entity combinations (e.g., "[location]'
            ' settlers")\n'
            '   For "who" questions about roles, focus on '
            "role/position terms. Be specific to context."
        )

        client = get_openai_client()
        response = client.chat.completions.create(
            model=QUERY_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
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
                                "enum": [
                                    "who",
                                    "when",
                                    "where",
                                    "person",
                                    "company",
                                    "factoid",
                                    "general",
                                ],
                            },
                            "person": {"type": "string"},
                            "entities": {
                                "type": "object",
                                "properties": {
                                    "persons": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "places": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "events": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "dates": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "families": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "companies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "industries": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "settlement_terms": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "persons",
                                    "places",
                                    "events",
                                    "dates",
                                    "families",
                                    "companies",
                                    "industries",
                                    "settlement_terms",
                                ],
                                "additionalProperties": False,
                            },
                            "expansions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query_type", "person", "entities", "expansions"],
                        "additionalProperties": False,
                    },
                },
            },
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
        logger.error("Error in query analysis: %s, using defaults", e)
        return default_result


def hybrid_search(
    query: str,
    k: int = 200,
    document_id: str | None = None,
    semantic_ratio: float = 0.75,
) -> list[tuple[str, float]]:
    """Run hybrid (keyword + semantic) search via Meilisearch."""
    try:
        client = get_meili_client()
        index = client.index("passages")

        search_params = {
            "hybrid": {"semanticRatio": semantic_ratio, "embedder": "default"},
            "limit": k,
            "attributesToRetrieve": ["id"],
            "showRankingScore": True,
        }

        if document_id:
            try:
                uuid.UUID(document_id)
            except (ValueError, AttributeError) as e:
                msg = f"Invalid document_id: {document_id}"
                raise ValueError(msg) from e
            search_params["filter"] = f"document_id = '{document_id}'"

        response = index.search(query, search_params)

        return [(hit["id"], hit.get("_rankingScore", 0.0)) for hit in response["hits"]]

    except Exception as e:
        logger.warning("Hybrid search failed: %s", e)
        return []


def _get_reranker():
    """Lazy-load the cross-encoder re-ranker model."""
    if not hasattr(_get_reranker, "_model"):
        from sentence_transformers import CrossEncoder

        _get_reranker._model = CrossEncoder(RERANKER_MODEL)  # noqa: SLF001
    return _get_reranker._model  # noqa: SLF001


def rerank_candidates(
    candidates: list[tuple[str, float]], query: str, conn, k: int = 20
) -> list[tuple[str, float]]:
    """Re-rank candidates using a cross-encoder model."""
    if not candidates:
        return candidates

    # Fetch passage texts for all candidates in one query
    passage_ids = [pid for pid, _ in candidates]
    placeholders = ",".join(["%s"] * len(passage_ids))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT id, text FROM passages WHERE id IN ({placeholders})
        """,  # noqa: S608
            passage_ids,
        )
        id_to_text = {row["id"]: row["text"] for row in cur.fetchall()}

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
    ranked = sorted(
        zip(valid_ids, scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(pid, float(score)) for pid, score in ranked[:k]]


def get_passage_details(conn, passage_ids: list[str]) -> list[dict[str, Any]]:
    """Get detailed passage information."""
    if not passage_ids:
        return []

    placeholders = ",".join(["%s"] * len(passage_ids))

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT p.id, p.text, p.page, d.title, p.headings_path
            FROM passages p
            JOIN documents d ON p.document_id = d.id
            WHERE p.id IN ({placeholders})
        """,  # noqa: S608
            passage_ids,
        )

        results = cur.fetchall()

        # Preserve the order of passage_ids
        id_to_result = {result["id"]: result for result in results}
        return [
            id_to_result[passage_id]
            for passage_id in passage_ids
            if passage_id in id_to_result
        ]


def format_results(results: list[dict[str, Any]], scores: dict[str, float]) -> str:
    """Format search results for display."""
    output = []

    # Check if we have multiple documents
    document_titles = {result.get("title", "Unknown") for result in results}
    multi_document = len(document_titles) > 1

    for i, result in enumerate(results, 1):
        passage_id = result["id"]
        score = scores.get(passage_id, 0.0)
        title = result["title"]
        page = result["page"]

        # Truncate text for snippet
        text = result["text"]
        snippet = text[:200] + "..." if len(text) > 200 else text

        # Show document title more prominently in multi-document scenarios
        if multi_document:
            output.append(f'{i:2d}  {score:.3f}  [{title}]  p. {page}  "{snippet}"')
        else:
            output.append(f'{i:2d}  {score:.3f}  {title}  p. {page}  "{snippet}"')

    return "\n".join(output)


def generate_answer(query: str, results: list[dict[str, Any]]) -> str:
    """Generate an LLM-based answer from search results."""
    if not results:
        return "I couldn't find any relevant information to answer your question."

    try:
        from tome.search.answer_generator import answer_query

        return answer_query(query, results)
    except Exception:
        top_result = results[0]
        title = top_result.get("title", "Unknown Document")
        return (
            f"Top result: {top_result['text'][:300]}..."
            f" (Source: {title}, p. {top_result['page']})"
        )


def search_codex(
    query: str, k: int = 20, document_id: str | None = None
) -> dict[str, Any]:
    """
    Core search pipeline: hybrid search → rerank → answer generation.

    Returns dict with keys: query_type, query_analysis, answer, results (list of
    dicts with id, text, page, title, headings_path, score).
    """
    query_analysis = analyze_query(query)
    query_type = query_analysis["query_type"]

    candidates = hybrid_search(query, k=200, document_id=document_id)
    if not candidates:
        return {
            "query_type": query_type,
            "query_analysis": query_analysis,
            "answer": "No candidates found.",
            "results": [],
        }

    with db_connection() as conn:
        reranked = rerank_candidates(candidates, query, conn, k=k)

        passage_ids = [pid for pid, _ in reranked]
        passage_details = get_passage_details(conn, passage_ids)
        score_map = dict(reranked)

        # Attach scores to results
        for detail in passage_details:
            detail["score"] = score_map.get(detail["id"], 0.0)

        answer = generate_answer(query, passage_details)

    return {
        "query_type": query_type,
        "query_analysis": query_analysis,
        "answer": answer,
        "results": passage_details,
    }


@click.command()
@click.option("--q", "query", required=True, help="Search query")
@click.option("--k", default=20, type=int, help="Number of results to return")
@click.option("--doc", "document_id", help="Filter by document ID")
def main(query: str, k: int, document_id: str | None):
    """Search passages using hybrid search with cross-encoder re-ranking."""
    configure_logging()

    logger.info("Searching for: %s", query)

    result = search_codex(query, k=k, document_id=document_id)
    logger.info("Query type: %s", result["query_type"])

    click.echo("\nAnswer:")
    click.echo("=" * 80)
    click.echo(result["answer"])

    click.echo(f"\nTop {len(result['results'])} results:")
    click.echo("=" * 80)
    scores = {r["id"]: r["score"] for r in result["results"]}
    click.echo(format_results(result["results"], scores))


if __name__ == "__main__":
    main()
