from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.document_loader import (
    DOCUMENT_DIRECTORY,
    load_all_documents,
    load_document,
    parse_front_matter,
    split_markdown_sections,
    split_section_content,
)


def test_parses_yaml_front_matter() -> None:
    text = """
---
document_id: DOC-TEST-001
document_type: test_document
title: Test Document
effective_date: 2025-01-01
tags:
  - test
  - example
---

# Test Document

## Summary

This is a test.
""".strip()

    metadata, body = parse_front_matter(text)

    assert metadata["document_id"] == (
        "DOC-TEST-001"
    )
    assert metadata["title"] == (
        "Test Document"
    )
    assert metadata["effective_date"] == (
        "2025-01-01"
    )
    assert metadata["tags"] == [
        "test",
        "example",
    ]
    assert "# Test Document" in body


def test_splits_markdown_by_section() -> None:
    body = """
# Example

## First Section

First section content.

## Second Section

Second section content.
""".strip()

    sections = split_markdown_sections(body)

    section_names = [
        section_name
        for section_name, _ in sections
    ]

    assert "First Section" in section_names
    assert "Second Section" in section_names


def test_splits_large_section_content() -> None:
    content = "\n\n".join(
        [
            "Paragraph one " * 20,
            "Paragraph two " * 20,
            "Paragraph three " * 20,
        ]
    )

    chunks = split_section_content(
        content=content,
        max_chars=300,
        overlap_chars=40,
    )

    assert len(chunks) > 1
    assert all(
        chunk.strip()
        for chunk in chunks
    )


def test_loads_campaign_document() -> None:
    document_path = (
        DOCUMENT_DIRECTORY
        / "sparkle-summer-2025-campaign-brief.md"
    )

    chunks = load_document(document_path)

    assert chunks

    first_chunk = chunks[0]

    assert first_chunk.document_id == (
        "DOC-CAMPAIGN-001"
    )

    assert first_chunk.title == (
        "Sparkle Summer 2025 Campaign Brief"
    )

    assert first_chunk.document_type == (
        "campaign_brief"
    )

    assert first_chunk.source_path.endswith(
        "sparkle-summer-2025-campaign-brief.md"
    )


def test_chunks_contain_citations() -> None:
    document_path = (
        DOCUMENT_DIRECTORY
        / "inventory-exception-report.md"
    )

    chunks = load_document(document_path)

    for chunk in chunks:
        assert chunk.citation
        assert chunk.title in chunk.citation
        assert chunk.section in chunk.citation


def test_loads_all_generated_documents() -> None:
    chunks = load_all_documents()

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    assert len(document_ids) == 7

    assert "DOC-CAMPAIGN-001" in document_ids
    assert "DOC-REVIEW-001" in document_ids
    assert "DOC-INVENTORY-001" in document_ids
    assert "DOC-POLICY-001" in document_ids


def test_overlapping_entities_exist_in_chunks() -> None:
    chunks = load_all_documents()

    matching_documents = {
        chunk.document_id
        for chunk in chunks
        if "FG-DW-LEM-500" in chunk.content
    }

    assert len(matching_documents) >= 3


def test_missing_document_raises_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_document(
            Path(
                "data/documents/"
                "missing-document.md"
            )
        )


def test_rejects_invalid_chunk_configuration() -> None:
    with pytest.raises(ValueError):
        split_section_content(
            content="Test content",
            max_chars=100,
            overlap_chars=100,
        )