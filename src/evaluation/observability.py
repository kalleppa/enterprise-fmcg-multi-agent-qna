from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator
from uuid import uuid4

from src.agents.coding_agent import CodingAnalysisAgent
from src.agents.document_agent import DocumentRetrievalAgent
from src.agents.internet_agent import InternetSearchAgent
from src.agents.orchestrator import (
    EnterpriseQnAOrchestrator,
    OrchestratorResponse,
)
from src.agents.router import IntentRouter
from src.agents.structured_agent import StructuredDataAgent
from src.evaluation.response_evaluator import (
    ResponseEvaluator,
)


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


@dataclass(frozen=True)
class TraceEvent:
    """One timed step within a request trace."""

    event_id: str
    name: str
    category: str
    method: str | None
    status: str
    started_at_utc: str
    duration_ms: float
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    """End-to-end observability trace."""

    trace_id: str
    question: str
    started_at_utc: str

    route: str | None = None
    status: str | None = None
    completed_at_utc: str | None = None
    duration_ms: float | None = None

    events: list[TraceEvent] = field(
        default_factory=list
    )

    tool_call_count: int = 0
    retry_count: int = 0
    fallback_count: int = 0

    model_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    retry_reasons: list[str] = field(
        default_factory=list
    )

    _started_timer: float = field(
        default_factory=time.perf_counter,
        repr=False,
    )

    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @contextmanager
    def span(
        self,
        name: str,
        category: str,
        method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Measure one execution step."""

        event_id = (
            f"event-{uuid4().hex[:12]}"
        )

        started_at = utc_now()
        start_timer = time.perf_counter()

        event_status = "success"
        event_error: str | None = None

        try:
            yield

        except Exception as error:
            event_status = "error"
            event_error = str(error)
            raise

        finally:
            duration_ms = (
                time.perf_counter()
                - start_timer
            ) * 1_000

            event = TraceEvent(
                event_id=event_id,
                name=name,
                category=category,
                method=method,
                status=event_status,
                started_at_utc=started_at,
                duration_ms=round(
                    duration_ms,
                    3,
                ),
                metadata=metadata or {},
                error=event_error,
            )

            with self._lock:
                self.events.append(event)

    def record_tool_call(self) -> None:
        with self._lock:
            self.tool_call_count += 1

    def record_retry(
        self,
        reason: str,
        fallback: bool = False,
    ) -> None:
        with self._lock:
            self.retry_count += 1
            self.retry_reasons.append(
                reason
            )

            if fallback:
                self.fallback_count += 1

    def record_model_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        """Record future LLM usage when model calls are added."""

        with self._lock:
            self.model_call_count += 1
            self.input_tokens += max(
                input_tokens,
                0,
            )
            self.output_tokens += max(
                output_tokens,
                0,
            )
            self.estimated_cost_usd += max(
                estimated_cost_usd,
                0.0,
            )

    def finish(
        self,
        route: str,
        status: str,
    ) -> None:
        with self._lock:
            self.route = route
            self.status = status
            self.completed_at_utc = utc_now()

            self.duration_ms = round(
                (
                    time.perf_counter()
                    - self._started_timer
                )
                * 1_000,
                3,
            )

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            agent_latency: dict[
                str,
                float,
            ] = defaultdict(float)

            agent_calls: dict[
                str,
                int,
            ] = defaultdict(int)

            for event in self.events:
                if event.category == "agent":
                    agent_latency[
                        event.name
                    ] += event.duration_ms

                    agent_calls[
                        event.name
                    ] += 1

            return {
                "trace_id": self.trace_id,
                "question": self.question,
                "route": self.route,
                "status": self.status,
                "started_at_utc": (
                    self.started_at_utc
                ),
                "completed_at_utc": (
                    self.completed_at_utc
                ),
                "duration_ms": self.duration_ms,
                "tool_call_count": (
                    self.tool_call_count
                ),
                "retry_count": self.retry_count,
                "fallback_count": (
                    self.fallback_count
                ),
                "retry_reasons": list(
                    self.retry_reasons
                ),
                "model_usage": {
                    "model_call_count": (
                        self.model_call_count
                    ),
                    "input_tokens": (
                        self.input_tokens
                    ),
                    "output_tokens": (
                        self.output_tokens
                    ),
                    "estimated_cost_usd": round(
                        self.estimated_cost_usd,
                        8,
                    ),
                },
                "per_agent_latency_ms": {
                    key: round(
                        value,
                        3,
                    )
                    for key, value
                    in agent_latency.items()
                },
                "per_agent_call_count": dict(
                    agent_calls
                ),
                "events": [
                    event.to_dict()
                    for event in self.events
                ],
            }


class InMemoryTraceStore:
    """Bounded in-memory trace storage."""

    def __init__(
        self,
        max_traces: int = 1_000,
    ) -> None:
        if max_traces <= 0:
            raise ValueError(
                "max_traces must be greater than zero."
            )

        self.max_traces = max_traces
        self._traces: dict[
            str,
            dict[str, Any],
        ] = {}

        self._trace_order: list[str] = []
        self._lock = threading.RLock()

    def add(
        self,
        trace: ExecutionTrace,
    ) -> None:
        trace_dict = trace.to_dict()

        with self._lock:
            self._traces[
                trace.trace_id
            ] = trace_dict

            self._trace_order.append(
                trace.trace_id
            )

            while (
                len(self._trace_order)
                > self.max_traces
            ):
                removed_id = (
                    self._trace_order.pop(0)
                )

                self._traces.pop(
                    removed_id,
                    None,
                )

    def get(
        self,
        trace_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(
                trace_id
            )

            return (
                dict(trace)
                if trace
                else None
            )

    def list_recent(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        with self._lock:
            trace_ids = self._trace_order[
                -limit:
            ]

            return [
                dict(
                    self._traces[trace_id]
                )
                for trace_id
                in reversed(trace_ids)
            ]


class ObservedAgentProxy:
    """Measure calls to a specialist agent."""

    OBSERVED_METHODS = {
        "answer",
        "analyze",
        "search",
    }

    def __init__(
        self,
        target: Any,
        agent_name: str,
        trace: ExecutionTrace,
    ) -> None:
        self._target = target
        self._agent_name = agent_name
        self._trace = trace

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        attribute = getattr(
            self._target,
            name,
        )

        if (
            not callable(attribute)
            or name
            not in self.OBSERVED_METHODS
        ):
            return attribute

        def measured_call(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            self._trace.record_tool_call()

            with self._trace.span(
                name=self._agent_name,
                category="agent",
                method=name,
            ):
                return attribute(
                    *args,
                    **kwargs,
                )

        return measured_call


class ObservedOrchestrator:
    """
    Execute the standard orchestrator with request-scoped
    instrumentation.
    """

    def __init__(
        self,
        router: IntentRouter | None = None,
        structured_agent: StructuredDataAgent | None = None,
        document_agent: DocumentRetrievalAgent | None = None,
        coding_agent: CodingAnalysisAgent | None = None,
        internet_agent: InternetSearchAgent | None = None,
        evaluator: ResponseEvaluator | None = None,
        trace_store: InMemoryTraceStore | None = None,
    ) -> None:
        self.router = router or IntentRouter()

        self.structured_agent = (
            structured_agent
            or StructuredDataAgent()
        )

        self.document_agent = (
            document_agent
            or DocumentRetrievalAgent()
        )

        self.coding_agent = (
            coding_agent
            or CodingAnalysisAgent()
        )

        self.internet_agent = (
            internet_agent
            or InternetSearchAgent()
        )

        self.evaluator = (
            evaluator
            or ResponseEvaluator()
        )

        self.trace_store = (
            trace_store
            or InMemoryTraceStore()
        )

    def answer(
        self,
        question: str,
    ) -> OrchestratorResponse:
        """Execute one fully observed agent request."""

        trace = ExecutionTrace(
            trace_id=(
                f"trace-{uuid4().hex[:16]}"
            ),
            question=question,
            started_at_utc=utc_now(),
        )

        request_orchestrator = (
            EnterpriseQnAOrchestrator(
                router=self.router,
                structured_agent=ObservedAgentProxy(
                    target=(
                        self.structured_agent
                    ),
                    agent_name=(
                        "structured_agent"
                    ),
                    trace=trace,
                ),
                document_agent=ObservedAgentProxy(
                    target=(
                        self.document_agent
                    ),
                    agent_name=(
                        "document_agent"
                    ),
                    trace=trace,
                ),
                coding_agent=ObservedAgentProxy(
                    target=(
                        self.coding_agent
                    ),
                    agent_name=(
                        "coding_agent"
                    ),
                    trace=trace,
                ),
                internet_agent=ObservedAgentProxy(
                    target=(
                        self.internet_agent
                    ),
                    agent_name=(
                        "internet_agent"
                    ),
                    trace=trace,
                ),
            )
        )

        with trace.span(
            name="orchestrator",
            category="orchestrator",
            method="answer",
        ):
            response = (
                request_orchestrator.answer(
                    question
                )
            )

        self._record_fallbacks(
            trace=trace,
            response=response,
        )

        trace.finish(
            route=response.route,
            status=response.status,
        )

        response.observability = (
            trace.to_dict()
        )

        response.evaluation = (
            self.evaluator.evaluate(
                question=question,
                response=response.to_dict(),
            ).to_dict()
        )

        self.trace_store.add(
            trace
        )

        return response

    def get_trace(
        self,
        trace_id: str,
    ) -> dict[str, Any] | None:
        return self.trace_store.get(
            trace_id
        )

    def list_recent_traces(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.trace_store.list_recent(
            limit=limit
        )

    @staticmethod
    def _record_fallbacks(
        trace: ExecutionTrace,
        response: OrchestratorResponse,
    ) -> None:
        document_result = (
            response.document_result
            or {}
        )

        retrieval_mode = (
            document_result.get(
                "retrieval_mode"
            )
        )

        if retrieval_mode == (
            "keyword_fallback"
        ):
            trace.record_retry(
                reason=(
                    "Semantic document retrieval failed; "
                    "keyword retrieval was used."
                ),
                fallback=True,
            )

        limitations = response.limitations

        for limitation in limitations:
            normalized = (
                limitation.lower()
            )

            if (
                "fallback" in normalized
                or "used keyword retrieval"
                in normalized
            ):
                if trace.fallback_count == 0:
                    trace.record_retry(
                        reason=limitation,
                        fallback=True,
                    )