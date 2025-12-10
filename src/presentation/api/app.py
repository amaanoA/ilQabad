"""FastAPI application factory.

This module provides the main FastAPI application instance and configuration.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.presentation.api import dependencies
from src.presentation.api.routes import router


# Store init config for lifespan
_init_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    if _init_config.get("init_pipeline", False):
        dir_path = _init_config.get("models_dir") or Path("models")
        dependencies.init_pipeline(models_dir=dir_path)

    yield

    # Shutdown
    dependencies.reset_pipeline()


def create_app(
    title: str = "ilQabad Attendance API",
    version: str = "1.0.0",
    init_pipeline: bool = False,
    models_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title for documentation.
        version: API version string.
        init_pipeline: Whether to initialize the ML pipeline on startup.
        models_dir: Directory containing ONNX models (if init_pipeline=True).

    Returns:
        Configured FastAPI application instance.
    """
    global _init_config
    _init_config = {
        "init_pipeline": init_pipeline,
        "models_dir": models_dir,
    }

    app = FastAPI(
        title=title,
        version=version,
        description="Face recognition attendance system for Somali schools",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Include API routes
    app.include_router(router)

    # Custom exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with consistent JSON format."""
        if isinstance(exc.detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": str(exc.detail),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors with consistent JSON format."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "detail": str(exc),
            },
        )

    return app


# Default application instance (for uvicorn)
app = create_app()
