from __future__ import annotations

import pytest

from src.agents.document_agent import (
    DocumentRetrievalAgent,
)


@pytest.fixture(scope="module")
def agent() -> DocumentRetrievalAgent:
    return DocumentRetrievalAgent()


def test_lists_available_documents(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "What documents are available?"
    )

    assert response.status == "success"
    assert response.retrieval_mode == "metadata"
    assert len(response.evidence) == 7


def test_retrieves_campaign_risks(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "What risks were identified in the "
        "Sparkle Summer campaign brief?"
    )

    assert response.status == "success"
    assert response.evidence

    document_ids = {
        evidence.document_id
        for evidence in response.evidence
    }

    assert "DOC-CAMPAIGN-001" in document_ids

    assert (
        response.metadata_filters[
            "campaign_name"
        ]
        == "Sparkle Summer 2025"
    )

    assert (
        response.metadata_filters[
            "document_type"
        ]
        == "campaign_brief"
    )


def test_retrieves_inventory_evidence(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "What caused stockouts for "
        "FG-DW-LEM-500 in Karnataka?"
    )

    assert response.status == "success"
    assert response.evidence

    document_ids = {
        evidence.document_id
        for evidence in response.evidence
    }

    assert "DOC-INVENTORY-001" in document_ids

    assert (
        response.metadata_filters["sku_id"]
        == "FG-DW-LEM-500"
    )

    assert (
        response.metadata_filters["state"]
        == "Karnataka"
    )


def test_supports_explicit_metadata_filter(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        question=(
            "What does the policy say about "
            "discounts and margins?"
        ),
        metadata_filters={
            "document_type": "pricing_policy",
        },
    )

    assert response.status == "success"
    assert response.evidence

    assert all(
        evidence.document_type
        == "pricing_policy"
        for evidence in response.evidence
    )


def test_infers_effective_date_filter(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "Show FreshGlow reports after 2025-07-01"
    )

    assert response.status == "success"

    assert (
        response.metadata_filters[
            "effective_date_from"
        ]
        == "2025-07-01"
    )

    for evidence in response.evidence:
        assert (
            evidence.metadata[
                "effective_date"
            ]
            >= "2025-07-01"
        )


def test_returns_citations(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "Why was replenishment delayed?"
    )

    assert response.status == "success"
    assert response.citations

    for evidence in response.evidence:
        assert evidence.citation
        assert evidence.title in evidence.citation
        assert evidence.section in evidence.citation
        assert evidence.source_path.endswith(".md")


def test_limits_chunks_per_document(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "FreshGlow sales inventory campaign "
        "stockout distributor replenishment",
        top_k=10,
    )

    document_counts: dict[str, int] = {}

    for evidence in response.evidence:
        document_counts[
            evidence.document_id
        ] = (
            document_counts.get(
                evidence.document_id,
                0,
            )
            + 1
        )

    assert all(
        count <= 2
        for count in document_counts.values()
    )


def test_rejects_empty_question(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer("")

    assert response.status == "clarification"
    assert not response.evidence


def test_supports_keyword_or_hybrid_mode(
    agent: DocumentRetrievalAgent,
) -> None:
    response = agent.answer(
        "Product availability problems in Karnataka"
    )

    assert response.status == "success"

    assert response.retrieval_mode in {
        "hybrid",
        "keyword_fallback",
    }