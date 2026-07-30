from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.retrieval.keyword_search import (
    KeywordSearchIndex,
    KeywordSearchResult,
)
from src.retrieval.semantic_search import (
    SemanticSearchIndex,
    SemanticSearchResult,
)


@dataclass(frozen=True)
class HybridSearchResult:
    """One result produced by hybrid rank fusion."""

    chunk_id: str
    document_id: str
    title: str
    document_type: str
    section: str
    content: str
    snippet: str
    source_path: str
    citation: str
    hybrid_score: float
    keyword_score: float | None
    semantic_score: float | None
    keyword_rank: int | None
    semantic_rank: int | None
    retrieval_methods: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridSearchIndex:
    """
    Combine BM25 and semantic retrieval.

    Reciprocal-rank fusion is used because BM25 scores and cosine
    similarity scores are not directly comparable.
    """

    def __init__(
        self,
        keyword_index: KeywordSearchIndex,
        semantic_index: SemanticSearchIndex,
    ) -> None:
        self.keyword_index = keyword_index
        self.semantic_index = semantic_index

    @classmethod
    def from_default_indexes(
        cls,
    ) -> "HybridSearchIndex":
        """Load the default keyword and semantic indexes."""

        return cls(
            keyword_index=(
                KeywordSearchIndex.from_default_index()
            ),
            semantic_index=(
                SemanticSearchIndex.load()
            ),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
        keyword_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[HybridSearchResult]:
        """Run keyword and semantic searches and fuse their ranks."""

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if keyword_weight < 0:
            raise ValueError(
                "keyword_weight cannot be negative."
            )

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if keyword_weight == 0 and semantic_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be positive."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        candidate_count = max(
            top_k * 4,
            10,
        )

        keyword_results = self.keyword_index.search(
            query=query,
            top_k=candidate_count,
            metadata_filters=metadata_filters,
        )

        semantic_results = self.semantic_index.search(
            query=query,
            top_k=candidate_count,
            metadata_filters=metadata_filters,
        )

        combined: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            record = self._get_or_create_record(
                combined=combined,
                result=result,
            )

            record["hybrid_score"] += (
                keyword_weight
                / (rrf_k + rank)
            )

            record["keyword_score"] = result.score
            record["keyword_rank"] = rank
            record["methods"].add("keyword")

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            record = self._get_or_create_record(
                combined=combined,
                result=result,
            )

            record["hybrid_score"] += (
                semantic_weight
                / (rrf_k + rank)
            )

            record["semantic_score"] = result.score
            record["semantic_rank"] = rank
            record["methods"].add("semantic")

        sorted_records = sorted(
            combined.values(),
            key=lambda record: (
                -record["hybrid_score"],
                record["title"],
                record["chunk_id"],
            ),
        )

        return [
            HybridSearchResult(
                chunk_id=record["chunk_id"],
                document_id=record["document_id"],
                title=record["title"],
                document_type=record[
                    "document_type"
                ],
                section=record["section"],
                content=record["content"],
                snippet=record["snippet"],
                source_path=record["source_path"],
                citation=record["citation"],
                hybrid_score=round(
                    record["hybrid_score"],
                    8,
                ),
                keyword_score=record[
                    "keyword_score"
                ],
                semantic_score=record[
                    "semantic_score"
                ],
                keyword_rank=record[
                    "keyword_rank"
                ],
                semantic_rank=record[
                    "semantic_rank"
                ],
                retrieval_methods=tuple(
                    sorted(record["methods"])
                ),
                metadata=record["metadata"],
            )
            for record in sorted_records[:top_k]
        ]

    @staticmethod
    def _get_or_create_record(
        combined: dict[str, dict[str, Any]],
        result: KeywordSearchResult
        | SemanticSearchResult,
    ) -> dict[str, Any]:
        """Create a shared fusion record for one chunk."""

        if result.chunk_id not in combined:
            combined[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "title": result.title,
                "document_type": result.document_type,
                "section": result.section,
                "content": result.content,
                "snippet": result.snippet,
                "source_path": result.source_path,
                "citation": result.citation,
                "metadata": result.metadata,
                "hybrid_score": 0.0,
                "keyword_score": None,
                "semantic_score": None,
                "keyword_rank": None,
                "semantic_rank": None,
                "methods": set(),
            }

        return combined[result.chunk_id]


def main() -> None:
    """Run example hybrid searches."""

    index = HybridSearchIndex.from_default_indexes()

    examples = [
        (
            "Why did Sparkle Summer miss its "
            "sales-lift target?"
        ),
        (
            "Supply problems affecting FreshGlow "
            "campaign performance"
        ),
        (
            "कर्नाटक में FreshGlow stockout issue"
        ),
    ]

    for query in examples:
        print("=" * 80)
        print("Query:", query)

        results = index.search(
            query=query,
            top_k=5,
        )

        for result in results:
            print()
            print("Hybrid score:", result.hybrid_score)
            print(
                "Retrieval methods:",
                result.retrieval_methods,
            )
            print("Citation:", result.citation)
            print("Snippet:", result.snippet)


if __name__ == "__main__":
    main()