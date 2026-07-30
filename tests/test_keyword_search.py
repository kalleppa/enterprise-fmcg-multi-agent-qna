from __future__ import annotations

import pytest

from src.retrieval.document_loader import (
    load_all_documents,
)
from src.retrieval.keyword_search import (
    KeywordSearchIndex,
    tokenize,
)


@pytest.fixture(scope="module")
def search_index() -> KeywordSearchIndex:
    return KeywordSearchIndex(
        load_all_documents()
    )


def test_tokenizer_preserves_sku() -> None:
    tokens = tokenize(
        "Show FG-DW-LEM-500 stockouts"
    )

    assert "fg-dw-lem-500" in tokens
    assert "stockouts" in tokens


def test_finds_inventory_exception_document(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query=(
            "Karnataka stockout "
            "FG-DW-LEM-500"
        ),
        top_k=5,
    )

    document_ids = {
        result.document_id
        for result in results
    }

    assert "DOC-INVENTORY-001" in document_ids


def test_finds_campaign_risks(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query=(
            "Sparkle Summer planned "
            "sales lift risks"
        ),
        top_k=5,
    )

    document_ids = {
        result.document_id
        for result in results
    }

    assert "DOC-CAMPAIGN-001" in document_ids


def test_filters_by_document_type(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="sales lift inventory risk",
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


def test_filters_by_brand(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="product launch",
        top_k=10,
        metadata_filters={
            "brand": "FreshGlow",
        },
    )

    assert results

    assert all(
        str(
            result.metadata.get("brand")
        ).lower()
        == "freshglow"
        for result in results
    )


def test_filters_by_region(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="distributor inventory",
        top_k=10,
        metadata_filters={
            "region": "South Region",
        },
    )

    assert results

    assert all(
        result.document_id
        in {
            "DOC-CAMPAIGN-001",
            "DOC-REVIEW-001",
            "DOC-DISTRIBUTOR-001",
            "DOC-INVENTORY-001",
        }
        for result in results
    )


def test_filters_by_tag(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="discount margin",
        top_k=10,
        metadata_filters={
            "tags": "pricing",
        },
    )

    assert results

    assert all(
        "pricing"
        in [
            str(tag).lower()
            for tag in result.metadata.get(
                "tags",
                [],
            )
        ]
        for result in results
    )


def test_filters_by_effective_date(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="FreshGlow performance",
        top_k=10,
        metadata_filters={
            "effective_date_from": (
                "2025-07-01"
            ),
        },
    )

    assert results

    for result in results:
        assert (
            result.metadata["effective_date"]
            >= "2025-07-01"
        )


def test_filter_supports_multiple_values(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="campaign sales lift",
        top_k=10,
        metadata_filters={
            "document_type": [
                "campaign_brief",
                "quarterly_business_review",
            ],
        },
    )

    assert results

    assert all(
        result.document_type
        in {
            "campaign_brief",
            "quarterly_business_review",
        }
        for result in results
    )


def test_results_include_citations(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="replenishment delay",
        top_k=3,
    )

    assert results

    for result in results:
        assert result.citation
        assert result.title in result.citation
        assert result.section in result.citation
        assert result.source_path.endswith(".md")


def test_returns_empty_for_unknown_terms(
    search_index: KeywordSearchIndex,
) -> None:
    results = search_index.search(
        query="xyznonexistentterm",
        top_k=5,
    )

    assert results == []


def test_rejects_empty_query(
    search_index: KeywordSearchIndex,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        search_index.search("")