from __future__ import annotations

from src.retrieval.semantic_search import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    SemanticSearchIndex,
)


def main() -> None:
    print(
        "Building semantic document index with model:"
    )

    print(DEFAULT_MODEL_NAME)
    print()

    index = SemanticSearchIndex.build(
        model_name=DEFAULT_MODEL_NAME,
        batch_size=16,
    )

    index.save()

    print()
    print(
        f"Indexed {len(index.chunks)} document chunks."
    )

    print(
        "Embedding dimension:",
        index.embeddings.shape[1],
    )

    print(
        "Embeddings saved to:",
        DEFAULT_EMBEDDING_PATH,
    )

    print(
        "Metadata saved to:",
        DEFAULT_METADATA_PATH,
    )


if __name__ == "__main__":
    main()