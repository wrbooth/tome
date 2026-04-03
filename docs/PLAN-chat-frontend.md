# Chat Frontend Implementation Plan

## Phase 1: API Foundation ✅ COMPLETE

_Everything else depends on this._

1. ✅ **Enrich search response** — `POST /search` now returns full passage `text`, `headings_path`, and `query_analysis` (entities, expansions, person). `search_codex()` passes through the full `analyze_query()` result.

2. ✅ **Add document detail endpoint** — `GET /documents/{id}` exposes `get_document_stats()`. Returns 404 if not found.

3. ✅ **Add document upload endpoint** — `POST /documents/upload` accepts multipart PDF/TXT + metadata (title, authors, pub_year). Runs `ingest_document()` in background via `asyncio.create_task` + `asyncio.to_thread`. Status polling at `GET /documents/upload/{task_id}`.

4. ✅ **Add streaming search endpoint** — `POST /search/stream` using SSE via `StreamingResponse`. Events: `search_results` (immediate), `token` (LLM streaming), `done`, `error`. New `stream_answer_with_llm()` generator in `answer_generator.py`.

5. ✅ **CORS middleware** — Added for frontend dev compatibility.

6. ✅ **Tests** — 21 tests in `tests/test_api.py` covering all endpoints. Full suite: 324 tests passing.

## Phase 2: Minimal Viable Frontend ✅ COMPLETE

_The core loop: ask a question, get an answer, see sources._

0. ✅ **Fix streaming endpoint** — Replaced `list()` token buffering with `asyncio.Queue` bridge for true token-by-token streaming from sync generator to async SSE response.

1. ✅ **Add `/api/` prefix to all routes** — Moved all endpoints to `APIRouter(prefix="/api")` so API routes coexist cleanly with static file serving. All 21 tests updated.

5. ✅ **Static file serving + base layout** — FastAPI serves the SPA from `api/static/`. Shell layout with chat area and sidebar. Vanilla JS + Tailwind v4 CDN (no build step). CDN deps: `@tailwindcss/browser@4`, `marked.js`, `DOMPurify`.

6. ✅ **Chat input + streaming answer display** — `fetch()` + `ReadableStream` SSE parser (POST endpoint), `requestAnimationFrame` token batching, typing indicator (bouncing dots), streaming cursor, markdown rendering via `marked.parse()` + DOMPurify sanitization.

7. ✅ **Source citations** — Parses `[Document Title, Page X]` references in the answer and replaces them with superscript numbered citation links. Clicking a citation opens a slide-out source panel on the right (420px) showing the full passage text, with the active citation highlighted and scrolled into view.

8. ✅ **Source panel** — Source cards appear in a slide-out side panel (not below the answer). Cards show title, page, heading path, and full passage text. Cards are built after the LLM answer completes (on the `done` SSE event). Score data is tracked internally but not displayed in the UI.

## Phase 3: Document Management ✅ COMPLETE

_Users need to browse what's indexed and add new material._

9. ✅ **Document sidebar** — Lists all documents from `GET /api/documents` with title/author/year. Click to filter searches to that document. Active filter shown as a dismissible chip above the chat input. Includes a search/filter input for the document list itself.

10. ✅ **Document detail view** — Click the info button on a document to see stats (passage count, page range, entity count, year count). Includes a "Filter searches to this document" button and a "Delete document" button with confirmation dialog.

11. ✅ **Document upload UI** — Upload button in sidebar opens a modal with file picker (PDF/TXT), metadata form (title, authors, pub year). Supports drag-and-drop. Progress indicator polls the status endpoint. Success refreshes the sidebar document list.

## Phase 4: Polish & Power Features

_Makes it pleasant for sustained research use._

12. **Conversation history** — ⚠️ Partial: Chat messages accumulate in the DOM during a session and a "New Chat" button clears the conversation. No cross-session persistence.

13. **Query type indicator** — ❌ Not implemented. The `query_analysis` data (including `query_type`) is received from the API but not displayed in the UI.

14. **System stats display** — ❌ Not implemented. The `GET /api/stats` endpoint exists but the frontend does not call it. The sidebar footer shows only the document count from `GET /api/documents`.

15. **Adjustable result count** — ❌ Not implemented. No `k` parameter is sent with search requests (server default of 20 is used). No settings UI exists.

16. **Health status indicator** — ❌ Not implemented. The `GET /api/health` endpoint exists but the frontend does not ping it.

## Dependency Graph

```
Phase 1 (API)          Phase 2 (Core UI)       Phase 3 (Docs)         Phase 4 (Polish)
─────────────          ─────────────────        ──────────────         ────────────────
1. Enrich search ───→  6. Chat + streaming
4. Streaming    ───→   7. Clickable sources
                       8. Source panel
2. Doc detail   ──────────────────────────────→  9.  Doc sidebar
                                                 10. Doc detail view
3. Upload API   ──────────────────────────────→  11. Upload UI
5. Base layout ────→   6–8 (all UI)
                                                                        12–16 (independent)
```

Phases 1–3 are complete. Phase 4 items are all independent and can be done in any order.
