from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.document_loader import (
    PROJECT_ROOT,
    load_all_documents,
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "data" / "generated"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "document_chunks.jsonl"
)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_all_documents()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for chunk in chunks:
            output_file.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    document_count = len(
        {
            chunk.document_id
            for chunk in chunks
        }
    )

    print(
        f"Indexed {document_count} documents."
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print(
        "Saved index to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()