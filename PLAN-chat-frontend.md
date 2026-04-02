# Chat Frontend Implementation Plan

## Phase 1: API Foundation ✅ COMPLETE

_Everything else depends on this._

1. ✅ **Enrich search response** — `POST /search` now returns full passage `text`, `headings_path`, and `query_analysis` (entities, expansions, person). `search_codex()` passes through the full `analyze_query()` result.

2. ✅ **Add document detail endpoint** — `GET /documents/{id}` exposes `get_document_stats()`. Returns 404 if not found.

3. ✅ **Add document upload endpoint** — `POST /documents/upload` accepts multipart PDF/TXT + metadata (title, authors, pub_year). Runs `ingest_document()` in background via `asyncio.create_task` + `asyncio.to_thread`. Status polling at `GET /documents/upload/{task_id}`.

4. ✅ **Add streaming search endpoint** — `POST /search/stream` using SSE via `StreamingResponse`. Events: `search_results` (immediate), `token` (LLM streaming), `done`, `error`. New `stream_answer_with_llm()` generator in `answer_generator.py`.

5. ✅ **CORS middleware** — Added for frontend dev compatibility.

6. ✅ **Tests** — 21 tests in `tests/test_api.py` covering all endpoints. Full suite: 324 tests passing.

## Phase 2: Minimal Viable Frontend

_The core loop: ask a question, get an answer, see sources._

5. **Static file serving + base layout** — FastAPI serves the SPA. Shell layout with chat area and sidebar. Vanilla JS + Tailwind (no build step, keeps deployment trivial).

6. **Chat input + streaming answer display** — Text input, submit, stream the LLM answer into a chat bubble. This is the single most important UI element.

7. **Source citations as clickable expandables** — Parse `[Document Title, Page X]` references in the answer, make them clickable to expand the full passage text inline.

8. **Result cards below answer** — Collapsible passage cards showing title, page, heading path, score, and expandable full text.

## Phase 3: Document Management

_Users need to browse what's indexed and add new material._

9. **Document sidebar** — List all documents from `GET /documents`, show title/author/year. Click to filter searches to that document. Active filter shown as a dismissible chip above the chat input.

10. **Document detail view** — Click a document in the sidebar to see its stats (passage count, page range, entities, years).

11. **Document upload UI** — Upload button in sidebar, file picker (PDF/TXT), metadata form (title, authors, pub year), progress indicator polling the status endpoint. Success adds the document to the sidebar list.

## Phase 4: Polish & Power Features

_Makes it pleasant for sustained research use._

12. **Conversation history** — Scrollable chat history within a session. "New conversation" button to clear.

13. **Query type indicator** — Subtle badge on each answer showing how the system classified the query (who/when/where/etc.).

14. **System stats display** — Small footer or dashboard panel showing document count, passage count, embedding coverage from `GET /stats`. Also serves as a health indicator.

15. **Adjustable result count** — Settings gear or slider to control `k`. Default to 10 for the UI (less overwhelming than 20).

16. **Health status indicator** — Periodic `GET /health` ping, subtle green/red dot in the corner.

## Dependency Graph

```
Phase 1 (API)          Phase 2 (Core UI)       Phase 3 (Docs)         Phase 4 (Polish)
─────────────          ─────────────────        ──────────────         ────────────────
1. Enrich search ───→  6. Chat + streaming
4. Streaming    ───→   7. Clickable sources
                       8. Result cards
2. Doc detail   ──────────────────────────────→  9.  Doc sidebar
                                                 10. Doc detail view
3. Upload API   ──────────────────────────────→  11. Upload UI
5. Base layout ────→   6–8 (all UI)
                                                                        12–16 (independent)
```

Phases 1 and 2 are the critical path. Phase 3 can partially overlap with Phase 2 (sidebar is independent of chat). Phase 4 items are all independent and can be done in any order.
