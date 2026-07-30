from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.document_loader import (
    DocumentChunk,
    PROJECT_ROOT,
)
from src.retrieval.keyword_search import (
    create_snippet,
    flatten_metadata,
    load_chunks_from_jsonl,
    tokenize,
)


DEFAULT_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
)

DEFAULT_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "semantic_embeddings.npz"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "semantic_index.json"
)


@dataclass(frozen=True)
class SemanticSearchResult:
    """One citation-ready semantic search result."""

    chunk_id: str
    document_id: str
    title: str
    document_type: str
    section: str
    content: str
    snippet: str
    source_path: str
    citation: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_embedding_text(
    chunk: DocumentChunk,
) -> str:
    """Create the text sent to the embedding model."""

    selected_metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key
        in {
            "brand",
            "brands",
            "product",
            "products",
            "sku_id",
            "sku_ids",
            "region",
            "regions",
            "state",
            "states",
            "campaign_id",
            "campaign_name",
            "document_type",
            "tags",
            "effective_date",
        }
    }

    metadata_text = " ".join(
        flatten_metadata(selected_metadata)
    )

    return "\n".join(
        [
            f"Document: {chunk.title}",
            f"Type: {chunk.document_type}",
            f"Section: {chunk.section}",
            f"Metadata: {metadata_text}",
            f"Content: {chunk.content}",
        ]
    )


def parse_iso_date(
    value: Any,
) -> date | None:
    """Convert a filter value into a date."""

    if value is None:
        return None

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def normalize_filter_values(
    value: Any,
) -> set[str]:
    """Normalize one or more metadata filter values."""

    if isinstance(
        value,
        (list, tuple, set),
    ):
        raw_values = value
    else:
        raw_values = [value]

    return {
        str(item).strip().lower()
        for item in raw_values
        if item is not None
    }


def get_chunk_filter_values(
    chunk: DocumentChunk,
    key: str,
) -> set[str]:
    """Return searchable metadata values for one filter key."""

    field_mapping = {
        "document_type": (
            chunk.document_type,
        ),
        "document_id": (
            chunk.document_id,
        ),
        "title": (
            chunk.title,
        ),
        "section": (
            chunk.section,
        ),
        "brand": (
            chunk.metadata.get("brand"),
            chunk.metadata.get("brands"),
        ),
        "product": (
            chunk.metadata.get("product"),
            chunk.metadata.get("products"),
        ),
        "sku": (
            chunk.metadata.get("sku_id"),
            chunk.metadata.get("sku_ids"),
        ),
        "sku_id": (
            chunk.metadata.get("sku_id"),
            chunk.metadata.get("sku_ids"),
        ),
        "region": (
            chunk.metadata.get("region"),
            chunk.metadata.get("regions"),
        ),
        "state": (
            chunk.metadata.get("state"),
            chunk.metadata.get("states"),
        ),
        "campaign": (
            chunk.metadata.get("campaign_name"),
        ),
        "campaign_name": (
            chunk.metadata.get("campaign_name"),
        ),
        "campaign_id": (
            chunk.metadata.get("campaign_id"),
        ),
        "tag": (
            chunk.metadata.get("tags"),
        ),
        "tags": (
            chunk.metadata.get("tags"),
        ),
    }

    raw_values = field_mapping.get(
        key,
        (
            chunk.metadata.get(key),
        ),
    )

    flattened: list[str] = []

    for value in raw_values:
        flattened.extend(
            flatten_metadata(value)
        )

    return {
        item.strip().lower()
        for item in flattened
        if item.strip()
    }


def chunk_matches_filters(
    chunk: DocumentChunk,
    metadata_filters: dict[str, Any],
) -> bool:
    """
    Apply metadata filters.

    Different filter keys use AND logic.
    Multiple values under one key use OR logic.
    """

    for raw_key, expected_value in (
        metadata_filters.items()
    ):
        key = raw_key.lower().strip()

        if key in {
            "effective_date_from",
            "min_effective_date",
        }:
            chunk_date = parse_iso_date(
                chunk.metadata.get("effective_date")
            )

            expected_date = parse_iso_date(
                expected_value
            )

            if (
                chunk_date is None
                or expected_date is None
                or chunk_date < expected_date
            ):
                return False

            continue

        if key in {
            "effective_date_to",
            "max_effective_date",
        }:
            chunk_date = parse_iso_date(
                chunk.metadata.get("effective_date")
            )

            expected_date = parse_iso_date(
                expected_value
            )

            if (
                chunk_date is None
                or expected_date is None
                or chunk_date > expected_date
            ):
                return False

            continue

        available_values = get_chunk_filter_values(
            chunk=chunk,
            key=key,
        )

        expected_values = normalize_filter_values(
            expected_value
        )

        if not available_values & expected_values:
            return False

    return True


class SemanticSearchIndex:
    """Dense-vector search index for document chunks."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        model: SentenceTransformer,
        model_name: str,
    ) -> None:
        if not chunks:
            raise ValueError(
                "At least one document chunk is required."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        self.chunks = chunks
        self.embeddings = embeddings.astype(
            np.float32
        )
        self.model = model
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 16,
    ) -> "SemanticSearchIndex":
        """Generate semantic embeddings for all document chunks."""

        chunks = load_chunks_from_jsonl()

        model = SentenceTransformer(
            model_name
        )

        texts = [
            build_embedding_text(chunk)
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return cls(
            chunks=chunks,
            embeddings=np.asarray(
                embeddings,
                dtype=np.float32,
            ),
            model=model,
            model_name=model_name,
        )

    def save(
        self,
        embedding_path: Path = DEFAULT_EMBEDDING_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        """Save the vectors and index metadata."""

        embedding_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunk_ids = np.asarray(
            [
                chunk.chunk_id
                for chunk in self.chunks
            ],
            dtype=str,
        )

        np.savez_compressed(
            embedding_path,
            embeddings=self.embeddings,
            chunk_ids=chunk_ids,
        )

        index_metadata = {
            "model_name": self.model_name,
            "chunk_count": len(self.chunks),
            "embedding_dimension": int(
                self.embeddings.shape[1]
            ),
            "normalized_embeddings": True,
            "embedding_file": (
                embedding_path.name
            ),
        }

        metadata_path.write_text(
            json.dumps(
                index_metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        embedding_path: Path = DEFAULT_EMBEDDING_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> "SemanticSearchIndex":
        """Load an existing semantic index."""

        if not embedding_path.exists():
            raise FileNotFoundError(
                "Semantic embedding index is missing. "
                "Run `python -m "
                "scripts.build_semantic_index` first."
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Semantic index metadata is missing. "
                "Run `python -m "
                "scripts.build_semantic_index` first."
            )

        index_metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        model_name = str(
            index_metadata["model_name"]
        )

        chunks = load_chunks_from_jsonl()

        with np.load(
            embedding_path,
            allow_pickle=False,
        ) as archive:
            embeddings = np.asarray(
                archive["embeddings"],
                dtype=np.float32,
            )

            stored_chunk_ids = [
                str(item)
                for item in archive[
                    "chunk_ids"
                ].tolist()
            ]

        current_chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if stored_chunk_ids != current_chunk_ids:
            raise ValueError(
                "Document chunks have changed since the "
                "semantic index was generated. Rebuild it with "
                "`python -m scripts.build_semantic_index`."
            )

        model = SentenceTransformer(
            model_name
        )

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            model=model,
            model_name=model_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
        minimum_score: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Retrieve chunks using dense-vector similarity."""

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # Embeddings are normalized, so dot product is
        # equivalent to cosine similarity.
        similarity_scores = (
            self.embeddings @ query_embedding
        )

        candidate_results: list[
            tuple[float, DocumentChunk]
        ] = []

        filters = metadata_filters or {}

        for index, chunk in enumerate(self.chunks):
            if not chunk_matches_filters(
                chunk=chunk,
                metadata_filters=filters,
            ):
                continue

            score = float(
                similarity_scores[index]
            )

            if (
                minimum_score is not None
                and score < minimum_score
            ):
                continue

            candidate_results.append(
                (
                    score,
                    chunk,
                )
            )

        candidate_results.sort(
            key=lambda item: (
                -item[0],
                item[1].title,
                item[1].chunk_number,
            )
        )

        query_tokens = tokenize(query)

        return [
            SemanticSearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                document_type=chunk.document_type,
                section=chunk.section,
                content=chunk.content,
                snippet=create_snippet(
                    content=chunk.content,
                    query_tokens=query_tokens,
                ),
                source_path=chunk.source_path,
                citation=chunk.citation,
                score=round(score, 6),
                metadata=chunk.metadata,
            )
            for score, chunk
            in candidate_results[:top_k]
        ]


def main() -> None:
    """Run example semantic searches."""

    index = SemanticSearchIndex.load()

    examples = [
        "Why did the promotion fail to reach its target?",
        (
            "Which supply problems reduced campaign "
            "performance?"
        ),
        (
            "कर्नाटक में FreshGlow की उपलब्धता की "
            "समस्या क्या थी?"
        ),
    ]

    for query in examples:
        print("=" * 80)
        print("Query:", query)

        results = index.search(
            query=query,
            top_k=3,
        )

        for result in results:
            print()
            print("Score:", result.score)
            print("Citation:", result.citation)
            print("Snippet:", result.snippet)


if __name__ == "__main__":
    main()