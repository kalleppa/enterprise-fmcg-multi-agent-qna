from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from src.retrieval.hybrid_search import (
    HybridSearchIndex,
    HybridSearchResult,
)
from src.retrieval.keyword_search import (
    KeywordSearchIndex,
    KeywordSearchResult,
)


DocumentAgentStatus = Literal[
    "success",
    "clarification",
    "unsupported",
    "error",
]


@dataclass(frozen=True)
class DocumentEvidence:
    """Citation-ready evidence returned by document retrieval."""

    chunk_id: str
    document_id: str
    title: str
    document_type: str
    section: str
    content: str
    snippet: str
    citation: str
    source_path: str
    score: float
    retrieval_methods: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentAgentResponse:
    """Standard response produced by the document agent."""

    status: DocumentAgentStatus
    question: str
    message: str
    evidence: list[DocumentEvidence] = field(
        default_factory=list
    )
    citations: list[str] = field(
        default_factory=list
    )
    metadata_filters: dict[str, Any] = field(
        default_factory=dict
    )
    retrieval_mode: str | None = None
    limitations: list[str] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["evidence"] = [
            evidence.to_dict()
            for evidence in self.evidence
        ]

        return result


ENTITY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "brand": {
        "FreshGlow": (
            "freshglow",
            "fresh glow",
        ),
        "PureHome": (
            "purehome",
            "pure home",
        ),
        "NutriBite": (
            "nutribite",
            "nutri bite",
        ),
    },
    "region": {
        "South Region": (
            "south region",
            "southern region",
        ),
        "West Region": (
            "west region",
            "western region",
        ),
        "North Region": (
            "north region",
            "northern region",
        ),
        "East Region": (
            "east region",
            "eastern region",
        ),
    },
    "state": {
        "Karnataka": (
            "karnataka",
            "ka",
        ),
        "Tamil Nadu": (
            "tamil nadu",
            "tn",
        ),
        "Maharashtra": (
            "maharashtra",
            "mh",
        ),
        "Gujarat": (
            "gujarat",
            "gj",
        ),
        "Delhi": (
            "delhi",
            "dl",
        ),
        "Uttar Pradesh": (
            "uttar pradesh",
            "up",
        ),
        "West Bengal": (
            "west bengal",
            "wb",
        ),
        "Odisha": (
            "odisha",
            "orissa",
            "od",
        ),
    },
    "campaign_name": {
        "Sparkle Summer 2025": (
            "sparkle summer 2025",
            "sparkle summer",
        ),
        "NutriBite Digital Boost": (
            "nutribite digital boost",
            "digital boost",
        ),
        "PureHome Hygiene Week": (
            "purehome hygiene week",
            "hygiene week",
        ),
    },
    "sku_id": {
        "FG-DW-LEM-500": (
            "fg-dw-lem-500",
            "fg dw lem 500",
        ),
        "FG-DW-LEM-1L": (
            "fg-dw-lem-1l",
            "fg dw lem 1l",
        ),
        "FG-SC-LAV-500": (
            "fg-sc-lav-500",
            "fg sc lav 500",
        ),
        "FG-SC-LAV-1L": (
            "fg-sc-lav-1l",
            "fg sc lav 1l",
        ),
        "PH-HW-ALO-250": (
            "ph-hw-alo-250",
            "ph hw alo 250",
        ),
        "PH-HW-ALO-500": (
            "ph-hw-alo-500",
            "ph hw alo 500",
        ),
        "PH-BW-NEE-500": (
            "ph-bw-nee-500",
            "ph bw nee 500",
        ),
        "PH-HW-ROS-250": (
            "ph-hw-ros-250",
            "ph hw ros 250",
        ),
        "NB-GR-CHO-250": (
            "nb-gr-cho-250",
            "nb gr cho 250",
        ),
        "NB-GR-HON-500": (
            "nb-gr-hon-500",
            "nb gr hon 500",
        ),
        "NB-PB-CHO-6P": (
            "nb-pb-cho-6p",
            "nb pb cho 6p",
        ),
    },
}


DOCUMENT_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "campaign_brief": (
        "campaign brief",
        "campaign plan",
        "promotion brief",
    ),
    "quarterly_business_review": (
        "quarterly review",
        "business review",
        "category review",
        "qbr",
    ),
    "distributor_review": (
        "distributor review",
        "distributor report",
    ),
    "inventory_exception_report": (
        "inventory exception",
        "inventory report",
        "stockout report",
    ),
    "product_launch": (
        "product launch",
        "launch note",
    ),
    "pricing_policy": (
        "pricing policy",
        "discount policy",
        "promotion policy",
    ),
    "product_discontinuation": (
        "discontinuation notice",
        "discontinued product",
        "product discontinuation",
    ),
}


class DocumentRetrievalAgent:
    """Retrieve citation-ready evidence from enterprise documents."""

    def __init__(
        self,
        keyword_index: KeywordSearchIndex | None = None,
        hybrid_index: HybridSearchIndex | None = None,
    ) -> None:
        self._keyword_index = keyword_index
        self._hybrid_index = hybrid_index

    def answer(
        self,
        question: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> DocumentAgentResponse:
        """Retrieve document evidence for a user question."""

        cleaned_question = question.strip()

        if not cleaned_question:
            return DocumentAgentResponse(
                status="clarification",
                question=question,
                message=(
                    "Please provide the document question "
                    "you want me to investigate."
                ),
            )

        if top_k <= 0:
            return DocumentAgentResponse(
                status="error",
                question=question,
                message="top_k must be greater than zero.",
            )

        if self._is_document_metadata_question(
            cleaned_question
        ):
            return self._list_available_documents(
                question=question
            )

        inferred_filters = self._detect_filters(
            cleaned_question
        )

        combined_filters = {
            **inferred_filters,
            **(metadata_filters or {}),
        }

        search_query = self._build_search_query(
            question=cleaned_question,
            filters=combined_filters,
        )

        limitations: list[str] = []

        try:
            search_results = self._search_hybrid(
                query=search_query,
                top_k=max(top_k * 3, 10),
                metadata_filters=combined_filters,
            )

            retrieval_mode = "hybrid"

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as error:
            limitations.append(
                "Semantic retrieval was unavailable, so the "
                "agent used keyword retrieval only."
            )

            limitations.append(
                f"Semantic retrieval detail: {error}"
            )

            try:
                search_results = self._search_keyword(
                    query=search_query,
                    top_k=max(top_k * 3, 10),
                    metadata_filters=combined_filters,
                )

                retrieval_mode = "keyword_fallback"

            except Exception as keyword_error:
                return DocumentAgentResponse(
                    status="error",
                    question=question,
                    message=(
                        "The document retrieval request could "
                        "not be completed."
                    ),
                    metadata_filters=combined_filters,
                    limitations=limitations,
                    errors=[str(keyword_error)],
                )

        evidence = self._deduplicate_results(
            results=search_results,
            top_k=top_k,
        )

        if not evidence:
            return DocumentAgentResponse(
                status="success",
                question=question,
                message=(
                    "No document evidence matched the query "
                    "and selected metadata filters."
                ),
                metadata_filters=combined_filters,
                retrieval_mode=retrieval_mode,
                limitations=limitations,
            )

        citations = list(
            dict.fromkeys(
                item.citation
                for item in evidence
            )
        )

        return DocumentAgentResponse(
            status="success",
            question=question,
            message=(
                f"Retrieved {len(evidence)} citation-ready "
                "document evidence chunk(s)."
            ),
            evidence=evidence,
            citations=citations,
            metadata_filters=combined_filters,
            retrieval_mode=retrieval_mode,
            limitations=limitations,
        )

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, Any],
    ) -> list[HybridSearchResult]:
        if self._hybrid_index is None:
            self._hybrid_index = (
                HybridSearchIndex.from_default_indexes()
            )

        return self._hybrid_index.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

    def _search_keyword(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, Any],
    ) -> list[KeywordSearchResult]:
        if self._keyword_index is None:
            self._keyword_index = (
                KeywordSearchIndex.from_default_index()
            )

        return self._keyword_index.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

    def _list_available_documents(
        self,
        question: str,
    ) -> DocumentAgentResponse:
        keyword_index = self._keyword_index

        if keyword_index is None:
            keyword_index = (
                KeywordSearchIndex.from_default_index()
            )

            self._keyword_index = keyword_index

        documents: dict[str, dict[str, Any]] = {}

        for chunk in keyword_index.chunks:
            if chunk.document_id in documents:
                continue

            documents[chunk.document_id] = {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "document_type": chunk.document_type,
                "effective_date": (
                    chunk.metadata.get(
                        "effective_date"
                    )
                ),
                "source_path": chunk.source_path,
                "tags": chunk.metadata.get(
                    "tags",
                    [],
                ),
            }

        ordered_documents = sorted(
            documents.values(),
            key=lambda item: (
                str(
                    item.get(
                        "effective_date",
                        "",
                    )
                ),
                item["title"],
            ),
            reverse=True,
        )

        evidence = [
            DocumentEvidence(
                chunk_id=document["document_id"],
                document_id=document["document_id"],
                title=document["title"],
                document_type=document[
                    "document_type"
                ],
                section="Document metadata",
                content="",
                snippet=(
                    f"Effective date: "
                    f"{document.get('effective_date')}"
                ),
                citation=document["title"],
                source_path=document["source_path"],
                score=1.0,
                retrieval_methods=("metadata",),
                metadata=document,
            )
            for document in ordered_documents
        ]

        return DocumentAgentResponse(
            status="success",
            question=question,
            message=(
                f"Found {len(evidence)} available "
                "enterprise documents."
            ),
            evidence=evidence,
            citations=[
                item.citation
                for item in evidence
            ],
            retrieval_mode="metadata",
        )

    def _detect_filters(
        self,
        question: str,
    ) -> dict[str, Any]:
        normalized = self._normalize_text(question)
        filters: dict[str, Any] = {}

        for key, canonical_values in (
            ENTITY_ALIASES.items()
        ):
            for canonical_value, aliases in (
                canonical_values.items()
            ):
                if any(
                    self._contains_phrase(
                        normalized,
                        alias,
                    )
                    for alias in aliases
                ):
                    filters[key] = canonical_value
                    break

        for document_type, aliases in (
            DOCUMENT_TYPE_ALIASES.items()
        ):
            if any(
                self._contains_phrase(
                    normalized,
                    alias,
                )
                for alias in aliases
            ):
                filters[
                    "document_type"
                ] = document_type
                break

        date_from = self._detect_date_from(
            normalized
        )

        if date_from:
            filters[
                "effective_date_from"
            ] = date_from

        date_to = self._detect_date_to(
            normalized
        )

        if date_to:
            filters[
                "effective_date_to"
            ] = date_to

        return filters

    @staticmethod
    def _detect_date_from(
        question: str,
    ) -> str | None:
        explicit_match = re.search(
            r"\b(?:after|since|from)\s+"
            r"(20\d{2}-\d{2}-\d{2})\b",
            question,
        )

        if explicit_match:
            return explicit_match.group(1)

        year_match = re.search(
            r"\b(?:after|since|from)\s+(20\d{2})\b",
            question,
        )

        if year_match:
            return f"{year_match.group(1)}-01-01"

        return None

    @staticmethod
    def _detect_date_to(
        question: str,
    ) -> str | None:
        explicit_match = re.search(
            r"\b(?:before|until|to)\s+"
            r"(20\d{2}-\d{2}-\d{2})\b",
            question,
        )

        if explicit_match:
            return explicit_match.group(1)

        year_match = re.search(
            r"\b(?:before|until)\s+(20\d{2})\b",
            question,
        )

        if year_match:
            return f"{year_match.group(1)}-12-31"

        return None

    @staticmethod
    def _build_search_query(
        question: str,
        filters: dict[str, Any],
    ) -> str:
        """
        Preserve the original question and append normalized entities.

        This helps both exact keyword matching and semantic retrieval.
        """

        filter_values = [
            str(value)
            for key, value in filters.items()
            if not key.startswith(
                "effective_date"
            )
        ]

        return " ".join(
            dict.fromkeys(
                [
                    question,
                    *filter_values,
                ]
            )
        )

    @staticmethod
    def _deduplicate_results(
        results: list[
            HybridSearchResult
            | KeywordSearchResult
        ],
        top_k: int,
    ) -> list[DocumentEvidence]:
        """
        Deduplicate evidence by document and section.

        At most two chunks are retained from a single document so that
        one long document does not dominate the answer context.
        """

        evidence: list[DocumentEvidence] = []
        seen_sections: set[
            tuple[str, str]
        ] = set()
        per_document_count: dict[str, int] = {}

        for result in results:
            section_key = (
                result.document_id,
                result.section.lower(),
            )

            if section_key in seen_sections:
                continue

            current_document_count = (
                per_document_count.get(
                    result.document_id,
                    0,
                )
            )

            if current_document_count >= 2:
                continue

            if isinstance(
                result,
                HybridSearchResult,
            ):
                score = result.hybrid_score
                methods = result.retrieval_methods

            else:
                score = result.score
                methods = ("keyword",)

            evidence.append(
                DocumentEvidence(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    title=result.title,
                    document_type=(
                        result.document_type
                    ),
                    section=result.section,
                    content=result.content,
                    snippet=result.snippet,
                    citation=result.citation,
                    source_path=result.source_path,
                    score=float(score),
                    retrieval_methods=methods,
                    metadata=result.metadata,
                )
            )

            seen_sections.add(section_key)

            per_document_count[
                result.document_id
            ] = current_document_count + 1

            if len(evidence) >= top_k:
                break

        return evidence

    @staticmethod
    def _is_document_metadata_question(
        question: str,
    ) -> bool:
        normalized = (
            DocumentRetrievalAgent._normalize_text(
                question
            )
        )

        phrases = (
            "what documents are available",
            "which documents are available",
            "list documents",
            "show documents",
            "available document types",
            "what reports are available",
        )

        return any(
            phrase in normalized
            for phrase in phrases
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = value.lower()
        normalized = normalized.replace("_", " ")
        normalized = re.sub(
            r"[^\w\-]+",
            " ",
            normalized,
        )
        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        escaped = re.escape(
            phrase.lower()
        )

        return bool(
            re.search(
                rf"(?<!\w){escaped}(?!\w)",
                text,
            )
        )


def main() -> None:
    """Run document-agent examples."""

    agent = DocumentRetrievalAgent()

    questions = [
        (
            "What risks were identified in the "
            "Sparkle Summer campaign brief?"
        ),
        (
            "Why did FreshGlow have stockout problems "
            "in Karnataka?"
        ),
        (
            "What does the pricing policy say about "
            "discounts and margin?"
        ),
        (
            "कर्नाटक में FreshGlow की availability "
            "problem क्या थी?"
        ),
        "What documents are available?",
    ]

    for question in questions:
        response = agent.answer(
            question=question,
            top_k=5,
        )

        print("=" * 80)
        print("Question:", question)
        print("Status:", response.status)
        print("Message:", response.message)
        print(
            "Retrieval mode:",
            response.retrieval_mode,
        )
        print(
            "Filters:",
            response.metadata_filters,
        )

        for evidence in response.evidence:
            print()
            print("Citation:", evidence.citation)
            print("Score:", evidence.score)
            print(
                "Methods:",
                evidence.retrieval_methods,
            )
            print("Snippet:", evidence.snippet)

        if response.limitations:
            print(
                "Limitations:",
                response.limitations,
            )


if __name__ == "__main__":
    main()