from __future__ import annotations

import pytest

from src.retrieval.semantic_search import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_METADATA_PATH,
    SemanticSearchIndex,
)


@pytest.fixture(scope="module")
def semantic_index() -> SemanticSearchIndex:
    if (
        not DEFAULT_EMBEDDING_PATH.exists()
        or not DEFAULT_METADATA_PATH.exists()
    ):
        pytest.skip(
            "Semantic index is missing. Run "
            "`python -m scripts.build_semantic_index` first."
        )

    return SemanticSearchIndex.load()


def test_semantic_search_returns_results(
    semantic_index: SemanticSearchIndex,
) -> None:
    results = semantic_index.search(
        query=(
            "Why did the promotion fail to "
            "reach its target?"
        ),
        top_k=5,
    )

    assert results
    assert len(results) <= 5


def test_semantic_search_finds_campaign_evidence(
    semantic_index: SemanticSearchIndex,
) -> None:
    results = semantic_index.search(
        query=(
            "Inventory shortages reduced the "
            "marketing campaign performance"
        ),
        top_k=7,
    )

    document_ids = {
        result.document_id
        for result in results
    }

    assert document_ids & {
        "DOC-REVIEW-001",
        "DOC-INVENTORY-001",
        "DOC-CAMPAIGN-001",
        "DOC-DISTRIBUTOR-001",
    }


def test_semantic_search_supports_filters(
    semantic_index: SemanticSearchIndex,
) -> None:
    results = semantic_index.search(
        query="campaign risks and expected results",
        top_k=10,
        metadata_filters={
            "document_type": "campaign_brief",
        },
    )

    assert results

    assert all(
        result.document_type
        == "campaign_brief"
        for result in results
    )


def test_semantic_results_include_citations(
    semantic_index: SemanticSearchIndex,
) -> None:
    results = semantic_index.search(
        query="replenishment delay",
        top_k=3,
    )

    assert results

    for result in results:
        assert result.citation
        assert result.source_path.endswith(".md")
        assert result.content
        assert -1.0 <= result.score <= 1.0


def test_semantic_search_rejects_empty_query(
    semantic_index: SemanticSearchIndex,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        semantic_index.search("")