from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_DIRECTORY = PROJECT_ROOT / "data" / "documents"

DEFAULT_MAX_CHARS = 1_200
DEFAULT_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class DocumentChunk:
    """A citation-ready section of an enterprise document."""

    chunk_id: str
    document_id: str
    title: str
    document_type: str
    section: str
    content: str
    source_path: str
    citation: str
    chunk_number: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk into a JSON-serializable dictionary."""

        return asdict(self)


def make_json_safe(value: Any) -> Any:
    """Convert YAML values such as dates into JSON-safe values."""

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def parse_front_matter(
    text: str,
) -> tuple[dict[str, Any], str]:
    """
    Parse YAML front matter from a Markdown document.

    Expected format:

    ---
    document_id: DOC-001
    title: Example document
    ---
    # Document body
    """

    normalized_text = text.lstrip("\ufeff")

    if not normalized_text.startswith("---"):
        return {}, normalized_text.strip()

    lines = normalized_text.splitlines()

    if not lines or lines[0].strip() != "---":
        return {}, normalized_text.strip()

    closing_index: int | None = None

    for index, line in enumerate(
        lines[1:],
        start=1,
    ):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise ValueError(
            "Document front matter starts with '---' "
            "but has no closing delimiter."
        )

    metadata_text = "\n".join(
        lines[1:closing_index]
    )

    body = "\n".join(
        lines[closing_index + 1 :]
    ).strip()

    loaded_metadata = yaml.safe_load(
        metadata_text
    )

    if loaded_metadata is None:
        metadata: dict[str, Any] = {}

    elif not isinstance(loaded_metadata, dict):
        raise ValueError(
            "Document front matter must contain "
            "a YAML mapping."
        )

    else:
        metadata = {
            str(key): make_json_safe(value)
            for key, value in loaded_metadata.items()
        }

    return metadata, body


def split_markdown_sections(
    body: str,
) -> list[tuple[str, str]]:
    """
    Divide Markdown content using headings.

    Each returned item contains:

    - section title
    - section body
    """

    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    sections: list[tuple[str, str]] = []

    current_section = "Document Overview"
    current_lines: list[str] = []

    for line in body.splitlines():
        heading_match = heading_pattern.match(line)

        if heading_match:
            section_content = "\n".join(
                current_lines
            ).strip()

            if section_content:
                sections.append(
                    (
                        current_section,
                        section_content,
                    )
                )

            current_section = (
                heading_match.group(2).strip()
            )
            current_lines = []
            continue

        current_lines.append(line)

    final_content = "\n".join(
        current_lines
    ).strip()

    if final_content:
        sections.append(
            (
                current_section,
                final_content,
            )
        )

    return sections


def split_large_paragraph(
    paragraph: str,
    max_chars: int,
) -> list[str]:
    """Split a paragraph that is larger than the chunk limit."""

    words = paragraph.split()

    if not words:
        return []

    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(
            current_words + [word]
        )

        if (
            current_words
            and len(candidate) > max_chars
        ):
            parts.append(
                " ".join(current_words)
            )
            current_words = [word]

        else:
            current_words.append(word)

    if current_words:
        parts.append(
            " ".join(current_words)
        )

    return parts


def split_section_content(
    content: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """
    Split a section into smaller chunks while preserving paragraphs.

    A small amount of content from the previous chunk is included
    in the next chunk to preserve local context.
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be greater than zero."
        )

    if overlap_chars < 0:
        raise ValueError(
            "overlap_chars cannot be negative."
        )

    if overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be smaller than max_chars."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            content,
        )
        if paragraph.strip()
    ]

    expanded_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            expanded_paragraphs.append(paragraph)
        else:
            expanded_paragraphs.extend(
                split_large_paragraph(
                    paragraph,
                    max_chars=max_chars,
                )
            )

    chunks: list[str] = []
    current_parts: list[str] = []

    for paragraph in expanded_paragraphs:
        candidate_parts = (
            current_parts + [paragraph]
        )

        candidate = "\n\n".join(
            candidate_parts
        )

        if (
            current_parts
            and len(candidate) > max_chars
        ):
            completed_chunk = "\n\n".join(
                current_parts
            ).strip()

            chunks.append(completed_chunk)

            overlap = (
                completed_chunk[-overlap_chars:]
                if overlap_chars
                else ""
            ).strip()

            current_parts = []

            if overlap:
                current_parts.append(overlap)

            current_parts.append(paragraph)

        else:
            current_parts.append(paragraph)

    if current_parts:
        final_chunk = "\n\n".join(
            current_parts
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def build_citation(
    title: str,
    section: str,
    chunk_number: int,
) -> str:
    """Create a human-readable citation label."""

    return (
        f"{title} — {section} "
        f"(chunk {chunk_number})"
    )


def load_document(
    document_path: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Load one Markdown document and create section-aware chunks."""

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document does not exist: {document_path}"
        )

    if document_path.suffix.lower() != ".md":
        raise ValueError(
            "Only Markdown documents are supported "
            "in this prototype."
        )

    raw_text = document_path.read_text(
        encoding="utf-8"
    )

    metadata, body = parse_front_matter(
        raw_text
    )

    document_id = str(
        metadata.get(
            "document_id",
            document_path.stem,
        )
    )

    title = str(
        metadata.get(
            "title",
            document_path.stem.replace(
                "-",
                " ",
            ).title(),
        )
    )

    document_type = str(
        metadata.get(
            "document_type",
            "unknown",
        )
    )

    sections = split_markdown_sections(body)

    chunks: list[DocumentChunk] = []
    chunk_number = 1

    relative_path = document_path.relative_to(
        PROJECT_ROOT
    ).as_posix()

    for section_title, section_content in sections:
        section_chunks = split_section_content(
            content=section_content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for content in section_chunks:
            chunk_id = (
                f"{document_id}-"
                f"{chunk_number:04d}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    document_type=document_type,
                    section=section_title,
                    content=content,
                    source_path=relative_path,
                    citation=build_citation(
                        title=title,
                        section=section_title,
                        chunk_number=chunk_number,
                    ),
                    chunk_number=chunk_number,
                    metadata={
                        **metadata,
                        "section": section_title,
                    },
                )
            )

            chunk_number += 1

    return chunks


def load_all_documents(
    document_directory: Path = DOCUMENT_DIRECTORY,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Load every Markdown document from the document directory."""

    if not document_directory.exists():
        raise FileNotFoundError(
            "Document directory does not exist: "
            f"{document_directory}"
        )

    document_paths = sorted(
        document_directory.glob("*.md")
    )

    if not document_paths:
        raise FileNotFoundError(
            "No Markdown documents were found. Run "
            "`python scripts/generate_documents.py` first."
        )

    all_chunks: list[DocumentChunk] = []

    for document_path in document_paths:
        all_chunks.extend(
            load_document(
                document_path=document_path,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return all_chunks


def main() -> None:
    """Print a document-ingestion summary."""

    chunks = load_all_documents()

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    print(
        f"Loaded {len(document_ids)} documents."
    )

    print(
        f"Created {len(chunks)} document chunks."
    )

    print("\nFirst five chunks")
    print("-----------------")

    for chunk in chunks[:5]:
        print("Chunk ID:", chunk.chunk_id)
        print("Citation:", chunk.citation)
        print("Source:", chunk.source_path)
        print(
            "Content:",
            chunk.content[:200].replace(
                "\n",
                " ",
            ),
        )
        print()


if __name__ == "__main__":
    main()