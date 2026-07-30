from __future__ import annotations

import pytest

from src.retrieval.hybrid_search import (
    HybridSearchIndex,
)
from src.retrieval.semantic_search import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_METADATA_PATH,
)


@pytest.fixture(scope="module")
def hybrid_index() -> HybridSearchIndex:
    if (
        not DEFAULT_EMBEDDING_PATH.exists()
        or not DEFAULT_METADATA_PATH.exists()
    ):
        pytest.skip(
            "Semantic index is missing. Run "
            "`python -m scripts.build_semantic_index` first."
        )

    return HybridSearchIndex.from_default_indexes()


def test_hybrid_search_returns_ranked_results(
    hybrid_index: HybridSearchIndex,
) -> None:
    results = hybrid_index.search(
        query=(
            "Why did Sparkle Summer miss its "
            "sales-lift target?"
        ),
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    scores = [
        result.hybrid_score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_hybrid_search_finds_overlapping_evidence(
    hybrid_index: HybridSearchIndex,
) -> None:
    results = hybrid_index.search(
        query=(
            "FG-DW-LEM-500 Karnataka "
            "campaign stockout"
        ),
        top_k=7,
    )

    document_ids = {
        result.document_id
        for result in results
    }

    assert "DOC-INVENTORY-001" in document_ids

    assert document_ids & {
        "DOC-CAMPAIGN-001",
        "DOC-REVIEW-001",
        "DOC-DISTRIBUTOR-001",
    }


def test_hybrid_search_reports_methods(
    hybrid_index: HybridSearchIndex,
) -> None:
    results = hybrid_index.search(
        query="Sparkle Summer sales lift",
        top_k=5,
    )

    assert results

    for result in results:
        assert result.retrieval_methods

        assert set(
            result.retrieval_methods
        ).issubset(
            {
                "keyword",
                "semantic",
            }
        )


def test_hybrid_search_applies_metadata_filters(
    hybrid_index: HybridSearchIndex,
) -> None:
    results = hybrid_index.search(
        query="pricing discount margin policy",
        top_k=5,
        metadata_filters={
            "document_type": "pricing_policy",
        },
    )

    assert results

    assert all(
        result.document_type
        == "pricing_policy"
        for result in results
    )


def test_exact_sku_query_remains_retrievable(
    hybrid_index: HybridSearchIndex,
) -> None:
    results = hybrid_index.search(
        query="FG-DW-LEM-500",
        top_k=5,
    )

    assert results

    assert any(
        "FG-DW-LEM-500" in result.content
        or result.metadata.get(
            "sku_id"
        )
        == "FG-DW-LEM-500"
        or "FG-DW-LEM-500"
        in result.metadata.get(
            "sku_ids",
            [],
        )
        for result in results
    )


def test_hybrid_search_rejects_invalid_weights(
    hybrid_index: HybridSearchIndex,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        hybrid_index.search(
            query="campaign",
            keyword_weight=0,
            semantic_weight=0,
        )