from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path as FastAPIPath,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ClearHistoryResponse,
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
)
from src.database.metadata import (
    DATABASE_PATH,
    get_metadata_summary,
)
from src.memory.conversation_manager import (
    ConversationManager,
)
from src.retrieval.keyword_search import (
    DEFAULT_INDEX_PATH,
)
from src.retrieval.semantic_search import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_METADATA_PATH,
)


load_dotenv()

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHART_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "charts"
)


MetadataProvider = Callable[
    [],
    dict[str, Any],
]


def configure_logging() -> None:
    """Configure basic application logging."""

    logging.basicConfig(
        level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ).upper(),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def get_allowed_origins() -> list[str]:
    """Read CORS origins from environment configuration."""

    raw_origins = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        (
            "http://localhost:3000,"
            "http://localhost:8501"
        ),
    )

    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]


def verify_api_key(
    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key",
            description=(
                "Required only when AGENT_API_KEY "
                "is configured."
            ),
        ),
    ] = None,
) -> None:
    """
    Protect API endpoints when AGENT_API_KEY is configured.

    Development mode remains open when the environment variable
    is empty. Production deployments should always configure it.
    """

    expected_api_key = os.getenv(
        "AGENT_API_KEY",
        "",
    ).strip()

    if not expected_api_key:
        return

    if (
        not x_api_key
        or not secrets.compare_digest(
            x_api_key,
            expected_api_key,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={
                "WWW-Authenticate": "ApiKey",
            },
        )


def build_default_metadata() -> dict[str, Any]:
    """Return structured and document metadata."""

    metadata = get_metadata_summary()

    document_manager = ConversationManager()

    document_response = (
        document_manager
        .orchestrator
        .document_agent
        .answer(
            "What documents are available?"
        )
    )

    return {
        "structured": metadata,
        "documents": [
            evidence.to_dict()
            for evidence
            in document_response.evidence
        ],
    }


def create_app(
    conversation_manager: ConversationManager
    | None = None,
    metadata_provider: MetadataProvider
    | None = None,
) -> FastAPI:
    """Create the FastAPI application."""

    configure_logging()

    manager = (
        conversation_manager
        or ConversationManager()
    )

    resolved_metadata_provider = (
        metadata_provider
        or build_default_metadata
    )

    application = FastAPI(
        title="Enterprise FMCG Multi-Agent Q&A",
        description=(
            "Enterprise-style FMCG question-answering API "
            "with structured retrieval, document retrieval, "
            "internet search, controlled analytics, and "
            "multi-turn memory."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.state.conversation_manager = (
        manager
    )

    application.state.metadata_provider = (
        resolved_metadata_provider
    )

    allowed_origins = get_allowed_origins()

    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=[
                "GET",
                "POST",
                "DELETE",
            ],
            allow_headers=[
                "Content-Type",
                "X-API-Key",
            ],
        )

    @application.get(
        "/",
        tags=["service"],
    )
    def root() -> dict[str, str]:
        """Return service navigation information."""

        return {
            "service": (
                "Enterprise FMCG Multi-Agent Q&A"
            ),
            "version": "0.1.0",
            "health": "/health",
            "documentation": "/docs",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["service"],
    )
    def health() -> HealthResponse:
        """Return dependency availability information."""

        return HealthResponse(
            status="healthy",
            service=(
                "enterprise-fmcg-multi-agent-qna"
            ),
            version="0.1.0",
            database_available=(
                DATABASE_PATH.exists()
            ),
            document_index_available=(
                DEFAULT_INDEX_PATH.exists()
            ),
            semantic_index_available=(
                DEFAULT_EMBEDDING_PATH.exists()
                and DEFAULT_METADATA_PATH.exists()
            ),
            internet_search_configured=bool(
                os.getenv(
                    "TAVILY_API_KEY",
                    "",
                ).strip()
            ),
        )

    @application.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            401: {
                "model": ErrorResponse,
            },
            500: {
                "model": ErrorResponse,
            },
        },
        tags=["agent"],
        dependencies=[
            Depends(verify_api_key),
        ],
    )
    def chat(
        request: ChatRequest,
    ) -> ChatResponse:
        """Execute one single-turn or multi-turn request."""

        try:
            result = manager.answer(
                question=request.question,
                session_id=request.session_id,
            )

            return ChatResponse(
                **jsonable_encoder(
                    result.to_dict()
                )
            )

        except Exception as error:
            LOGGER.exception(
                "Chat request failed."
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "The agent could not complete "
                    "the request."
                ),
            ) from error

    @application.get(
        "/history/{session_id}",
        response_model=HistoryResponse,
        responses={
            401: {
                "model": ErrorResponse,
            },
            404: {
                "model": ErrorResponse,
            },
        },
        tags=["conversation"],
        dependencies=[
            Depends(verify_api_key),
        ],
    )
    def get_history(
        session_id: Annotated[
            str,
            FastAPIPath(
                min_length=1,
                max_length=200,
            ),
        ],
    ) -> HistoryResponse:
        """Return conversation history for one session."""

        history = manager.get_history(
            session_id
        )

        if (
            not history["recent_turns"]
            and not history[
                "compressed_summary"
            ]
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "Conversation session was not found."
                ),
            )

        return HistoryResponse(
            **jsonable_encoder(history)
        )

    @application.delete(
        "/history/{session_id}",
        response_model=ClearHistoryResponse,
        responses={
            401: {
                "model": ErrorResponse,
            },
        },
        tags=["conversation"],
        dependencies=[
            Depends(verify_api_key),
        ],
    )
    def clear_history(
        session_id: Annotated[
            str,
            FastAPIPath(
                min_length=1,
                max_length=200,
            ),
        ],
    ) -> ClearHistoryResponse:
        """Delete one conversation session."""

        cleared = manager.clear_history(
            session_id
        )

        return ClearHistoryResponse(
            session_id=session_id,
            cleared=cleared,
        )

    @application.get(
        "/metadata",
        responses={
            401: {
                "model": ErrorResponse,
            },
            500: {
                "model": ErrorResponse,
            },
        },
        tags=["metadata"],
        dependencies=[
            Depends(verify_api_key),
        ],
    )
    def metadata() -> dict[str, Any]:
        """Return available datasets, KPIs, dimensions, and documents."""

        try:
            result = resolved_metadata_provider()

            return jsonable_encoder(
                result
            )

        except Exception as error:
            LOGGER.exception(
                "Metadata retrieval failed."
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Metadata could not be retrieved. "
                    "Confirm that the database and document "
                    "index have been generated."
                ),
            ) from error

    @application.get(
        "/charts/{filename}",
        responses={
            401: {
                "model": ErrorResponse,
            },
            404: {
                "model": ErrorResponse,
            },
        },
        tags=["analysis"],
        dependencies=[
            Depends(verify_api_key),
        ],
    )
    def get_chart(
        filename: Annotated[
            str,
            FastAPIPath(
                min_length=1,
                max_length=255,
            ),
        ],
    ) -> FileResponse:
        """Return a generated PNG analysis chart."""

        safe_filename = Path(
            filename
        ).name

        if (
            safe_filename != filename
            or not safe_filename.lower().endswith(
                ".png"
            )
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail="Invalid chart filename.",
            )

        chart_directory = (
            CHART_DIRECTORY.resolve()
        )

        chart_path = (
            chart_directory
            / safe_filename
        ).resolve()

        if (
            chart_path.parent
            != chart_directory
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail="Invalid chart path.",
            )

        if not chart_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chart was not found.",
            )

        return FileResponse(
            path=chart_path,
            media_type="image/png",
            filename=safe_filename,
        )

    return application


app = create_app()