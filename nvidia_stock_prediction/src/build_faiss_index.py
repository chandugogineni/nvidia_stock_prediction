import os
import pickle

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


DOCUMENT_FILE = (
    "data/news/nvidia_documents.csv"
)

INDEX_FILE = (
    "data/vector_store/nvidia_news.faiss"
)

METADATA_FILE = (
    "data/vector_store/metadata.pkl"
)

MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


def build_index():

    os.makedirs(
        "data/vector_store",
        exist_ok=True
    )

    df = pd.read_csv(
        DOCUMENT_FILE
    )

    df["text"] = (
        df["text"]
        .fillna("")
    )

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = (
        df["text"]
        .tolist()
    )

    print(
        f"Creating embeddings for "
        f"{len(texts)} documents..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = (
        embeddings.astype(
            "float32"
        )
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    metadata = (
        df.to_dict(
            orient="records"
        )
    )

    with open(
        METADATA_FILE,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )

    print(
        "\nFAISS index created."
    )

    print(
        f"Documents: {len(texts)}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    print(
        f"Index: {INDEX_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    build_index()