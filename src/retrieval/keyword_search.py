from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.retrieval.document_loader import (
    DocumentChunk,
    PROJECT_ROOT,
    load_all_documents,
)


DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "document_chunks.jsonl"
)

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class KeywordSearchResult:
    """One citation-ready keyword-search result."""

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


def tokenize(text: str) -> list[str]:
    """
    Tokenize document and query text.

    Hyphenated values such as FG-DW-LEM-500 are retained as
    complete tokens and also split into component tokens.
    """

    tokens: list[str] = []

    for match in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(match)

        if "-" in match:
            tokens.extend(
                part
                for part in match.split("-")
                if part
            )

    return tokens


def flatten_metadata(value: Any) -> list[str]:
    """Convert nested metadata into searchable strings."""

    if value is None:
        return []

    if isinstance(value, dict):
        values: list[str] = []

        for key, item in value.items():
            values.append(str(key))
            values.extend(flatten_metadata(item))

        return values

    if isinstance(value, (list, tuple, set)):
        values = []

        for item in value:
            values.extend(flatten_metadata(item))

        return values

    return [str(value)]


def build_searchable_text(
    chunk: DocumentChunk,
) -> str:
    """
    Create weighted searchable text.

    Repeating the title and section gives them more influence than
    ordinary body text in BM25 term-frequency scoring.
    """

    metadata_text = " ".join(
        flatten_metadata(chunk.metadata)
    )

    return "\n".join(
        [
            chunk.title,
            chunk.title,
            chunk.title,
            chunk.section,
            chunk.section,
            chunk.content,
            metadata_text,
        ]
    )


def parse_iso_date(value: Any) -> date | None:
    """Parse an ISO-formatted metadata date."""

    if value is None:
        return None

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def create_snippet(
    content: str,
    query_tokens: Iterable[str],
    max_chars: int = 280,
) -> str:
    """Create a compact snippet around the first matching term."""

    clean_content = re.sub(
        r"\s+",
        " ",
        content,
    ).strip()

    if len(clean_content) <= max_chars:
        return clean_content

    lowered_content = clean_content.lower()

    match_positions = [
        lowered_content.find(token.lower())
        for token in query_tokens
        if lowered_content.find(token.lower()) >= 0
    ]

    if not match_positions:
        return clean_content[:max_chars].rstrip() + "..."

    first_match = min(match_positions)

    start = max(
        0,
        first_match - max_chars // 3,
    )

    end = min(
        len(clean_content),
        start + max_chars,
    )

    snippet = clean_content[start:end].strip()

    if start > 0:
        snippet = "..." + snippet

    if end < len(clean_content):
        snippet += "..."

    return snippet


def load_chunks_from_jsonl(
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[DocumentChunk]:
    """Load document chunks from the generated JSONL index."""

    if not index_path.exists():
        return load_all_documents()

    chunks: list[DocumentChunk] = []

    with index_path.open(
        "r",
        encoding="utf-8",
    ) as index_file:
        for line_number, line in enumerate(
            index_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in document index at "
                    f"line {line_number}: {error}"
                ) from error

            chunks.append(
                DocumentChunk(
                    chunk_id=record["chunk_id"],
                    document_id=record["document_id"],
                    title=record["title"],
                    document_type=record[
                        "document_type"
                    ],
                    section=record["section"],
                    content=record["content"],
                    source_path=record["source_path"],
                    citation=record["citation"],
                    chunk_number=int(
                        record["chunk_number"]
                    ),
                    metadata=record.get(
                        "metadata",
                        {},
                    ),
                )
            )

    return chunks


class KeywordSearchIndex:
    """In-memory BM25 index for enterprise documents."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError(
                "At least one document chunk is required."
            )

        if k1 <= 0:
            raise ValueError(
                "k1 must be greater than zero."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "b must be between zero and one."
            )

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.document_tokens: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        self.document_frequencies: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(
                build_searchable_text(chunk)
            )

            frequencies = Counter(tokens)

            self.document_tokens.append(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))

            for token in frequencies:
                self.document_frequencies[token] += 1

        self.document_count = len(chunks)

        self.average_document_length = (
            sum(self.document_lengths)
            / self.document_count
        )

    @classmethod
    def from_default_index(
        cls,
    ) -> "KeywordSearchIndex":
        """Build the index from JSONL or generated documents."""

        return cls(
            chunks=load_chunks_from_jsonl()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[KeywordSearchResult]:
        """
        Search document chunks.

        Filter behavior:

        - Different filter keys use AND logic.
        - Multiple values for one key use OR logic.
        - Matching is case-insensitive.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_frequency = Counter(query_tokens)

        scored_results: list[
            tuple[float, DocumentChunk]
        ] = []

        for index, chunk in enumerate(self.chunks):
            if not self._matches_filters(
                chunk,
                metadata_filters or {},
            ):
                continue

            score = self._calculate_score(
                document_index=index,
                query_frequency=query_frequency,
            )

            score += self._phrase_boost(
                chunk=chunk,
                query=query,
            )

            if score <= 0:
                continue

            scored_results.append(
                (
                    score,
                    chunk,
                )
            )

        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1].title,
                item[1].chunk_number,
            )
        )

        return [
            KeywordSearchResult(
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
            for score, chunk in scored_results[:top_k]
        ]

    def _calculate_score(
        self,
        document_index: int,
        query_frequency: Counter[str],
    ) -> float:
        """Calculate a BM25 score for one document chunk."""

        score = 0.0

        document_length = self.document_lengths[
            document_index
        ]

        frequencies = self.term_frequencies[
            document_index
        ]

        for token, query_count in (
            query_frequency.items()
        ):
            term_frequency = frequencies.get(
                token,
                0,
            )

            if term_frequency == 0:
                continue

            document_frequency = (
                self.document_frequencies[token]
            )

            inverse_document_frequency = math.log(
                1
                + (
                    self.document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            normalization = (
                1
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            term_score = (
                inverse_document_frequency
                * (
                    term_frequency
                    * (self.k1 + 1)
                )
                / (
                    term_frequency
                    + self.k1 * normalization
                )
            )

            query_weight = (
                1
                + math.log(query_count)
            )

            score += term_score * query_weight

        return score

    @staticmethod
    def _phrase_boost(
        chunk: DocumentChunk,
        query: str,
    ) -> float:
        """Boost exact phrase matches in important fields."""

        normalized_query = re.sub(
            r"\s+",
            " ",
            query.lower(),
        ).strip()

        if not normalized_query:
            return 0.0

        boost = 0.0

        if normalized_query in chunk.title.lower():
            boost += 4.0

        if normalized_query in chunk.section.lower():
            boost += 2.5

        if normalized_query in chunk.content.lower():
            boost += 2.0

        return boost

    def _matches_filters(
        self,
        chunk: DocumentChunk,
        metadata_filters: dict[str, Any],
    ) -> bool:
        """Return whether a chunk satisfies all filters."""

        for key, expected_value in (
            metadata_filters.items()
        ):
            normalized_key = key.lower().strip()

            if normalized_key in {
                "effective_date_from",
                "min_effective_date",
            }:
                chunk_date = parse_iso_date(
                    chunk.metadata.get(
                        "effective_date"
                    )
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

            if normalized_key in {
                "effective_date_to",
                "max_effective_date",
            }:
                chunk_date = parse_iso_date(
                    chunk.metadata.get(
                        "effective_date"
                    )
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

            available_values = self._get_filter_values(
                chunk=chunk,
                key=normalized_key,
            )

            expected_values = self._normalize_filter_values(
                expected_value
            )

            if not (
                available_values
                & expected_values
            ):
                return False

        return True

    @staticmethod
    def _normalize_filter_values(
        value: Any,
    ) -> set[str]:
        """Normalize one or many filter values."""

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

    @staticmethod
    def _get_filter_values(
        chunk: DocumentChunk,
        key: str,
    ) -> set[str]:
        """Read filter values from chunk fields and metadata."""

        field_mapping = {
            "document_type": (
                chunk.document_type,
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
            "sku_id": (
                chunk.metadata.get("sku_id"),
                chunk.metadata.get("sku_ids"),
            ),
            "sku": (
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
            "tag": (
                chunk.metadata.get("tags"),
            ),
            "tags": (
                chunk.metadata.get("tags"),
            ),
            "campaign_id": (
                chunk.metadata.get("campaign_id"),
            ),
            "campaign_name": (
                chunk.metadata.get(
                    "campaign_name"
                ),
            ),
            "campaign": (
                chunk.metadata.get(
                    "campaign_name"
                ),
            ),
            "document_id": (
                chunk.document_id,
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
            value.strip().lower()
            for value in flattened
            if value.strip()
        }


def main() -> None:
    """Run example keyword searches."""

    index = KeywordSearchIndex.from_default_index()

    examples = [
        (
            "Karnataka stockout FG-DW-LEM-500",
            {},
        ),
        (
            "Sparkle Summer planned sales lift risks",
            {
                "document_type": "campaign_brief",
            },
        ),
        (
            "replenishment delay",
            {
                "region": "South Region",
                "effective_date_from": "2025-01-01",
            },
        ),
    ]

    for query, filters in examples:
        print("=" * 80)
        print("Query:", query)
        print("Filters:", filters)

        results = index.search(
            query=query,
            top_k=3,
            metadata_filters=filters,
        )

        for result in results:
            print()
            print("Score:", result.score)
            print("Citation:", result.citation)
            print("Snippet:", result.snippet)


if __name__ == "__main__":
    main()