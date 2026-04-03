"""
Tome API Application

FastAPI app with search, document management, and streaming endpoints.
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from tome.api.routes.documents import router as documents_router
from tome.api.routes.search import router as search_router
from tome.api.routes.upload import router as upload_router
from tome.config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Tome Search API", version="2.0.0")


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Skip noisy static-file and health-check requests
        path = request.url.path
        if path.startswith("/static/") or path == "/api/health":
            return response

        logger.info(
            "%s %s → %d (%.0fms)",
            request.method,
            path,
            response.status_code,
            duration_ms,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)

# CORS -- allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include route modules
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(upload_router)


# ---------------------------------------------------------------------------
# Static file serving (SPA) — no-cache in development
# ---------------------------------------------------------------------------


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Add no-cache headers to static file responses for development."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheStaticMiddleware)

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for SPA routing."""
        return FileResponse(str(_static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
