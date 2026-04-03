"""Search API routes."""

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tome.config import db_connection
from tome.models import QueryAnalysisInfo, QueryEntities
from tome.search.answer_generator import stream_answer_with_llm
from tome.search.search import (
    analyze_query,
    get_passage_details,
    hybrid_search,
    rerank_candidates,
    search_codex,
)

logger = logging.getLogger(__name__)

# Sentinel for queue-based streaming bridge
_SENTINEL = object()

router = APIRouter(prefix="/api", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    k: int = 20
    document_id: str | None = None


class SearchResult(BaseModel):
    """A single search result with full passage details."""

    passage_id: str
    title: str
    page: int
    snippet: str
    text: str
    headings_path: list[str] = []
    score: float


class SearchResponse(BaseModel):
    query_type: str
    query_analysis: QueryAnalysisInfo
    answer: str
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search passages and generate an answer."""
    try:
        result = await asyncio.to_thread(
            search_codex, request.query, k=request.k, document_id=request.document_id
        )

        search_results = []
        for r in result["results"]:
            text = r["text"]
            snippet = text[:200] + "..." if len(text) > 200 else text
            search_results.append(
                SearchResult(
                    passage_id=r["id"],
                    title=r["title"],
                    page=r["page"],
                    snippet=snippet,
                    text=text,
                    headings_path=r.get("headings_path") or [],
                    score=r["score"],
                )
            )

        qa = result.get("query_analysis", {})
        query_analysis = QueryAnalysisInfo(
            query_type=qa.get("query_type", result["query_type"]),
            person=qa.get("person"),
            entities=QueryEntities(**qa.get("entities", {})),
            expansions=qa.get("expansions", []),
        )

        return SearchResponse(
            query_type=result["query_type"],
            query_analysis=query_analysis,
            answer=result["answer"],
            results=search_results,
        )

    except Exception as e:
        logger.exception("Search failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e)) from e


def _sse_event(event: str, data) -> str:
    """Format a Server-Sent Event string."""
    import json as _json

    payload = _json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/search/stream")
async def search_stream(request: SearchRequest):  # noqa: C901
    """Stream search results + LLM answer via Server-Sent Events.

    Events emitted:
      - search_results: immediate results + query analysis
      - token: individual LLM answer tokens
      - done: signals completion
      - error: on failure
    """

    async def _generate():  # noqa: C901
        try:
            qa = await asyncio.to_thread(analyze_query, request.query)

            candidates = await asyncio.to_thread(
                hybrid_search, request.query, 200, request.document_id
            )

            if not candidates:
                yield _sse_event(
                    "search_results",
                    {
                        "query_type": qa["query_type"],
                        "query_analysis": qa,
                        "results": [],
                    },
                )
                yield _sse_event("token", "No candidates found.")
                yield _sse_event("done", {})
                return

            def _rerank_and_detail():
                with db_connection() as conn:
                    reranked = rerank_candidates(
                        candidates, request.query, conn, k=request.k
                    )
                    passage_ids = [pid for pid, _ in reranked]
                    details = get_passage_details(conn, passage_ids)
                    score_map = dict(reranked)
                    for d in details:
                        d["score"] = score_map.get(d["id"], 0.0)
                    return details

            passage_details = await asyncio.to_thread(_rerank_and_detail)

            # Send search results immediately
            results_payload = []
            for r in passage_details:
                text = r["text"]
                results_payload.append(
                    {
                        "passage_id": r["id"],
                        "title": r["title"],
                        "page": r["page"],
                        "snippet": text[:200] + "..." if len(text) > 200 else text,
                        "text": text,
                        "headings_path": r.get("headings_path") or [],
                        "score": r["score"],
                    }
                )

            yield _sse_event(
                "search_results",
                {
                    "query_type": qa["query_type"],
                    "query_analysis": qa,
                    "results": results_payload,
                },
            )

            # Stream LLM answer tokens via queue bridge
            chunks = [
                {"text": r["text"], "page": r["page"], "title": r["title"]}
                for r in passage_details
            ]

            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _produce_tokens():
                try:
                    for token in stream_answer_with_llm(request.query, chunks):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            loop.run_in_executor(None, _produce_tokens)

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield _sse_event("token", item)

            yield _sse_event("done", {})

        except Exception as e:
            logger.exception("Streaming search failed")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
