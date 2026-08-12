import os
import pickle

import faiss
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("API KEY")
)


INPUT_FILE = "data/news/nvidia_documents.csv"

VECTOR_DIR = "data/vector_store"

INDEX_FILE = f"{VECTOR_DIR}/nvidia.faiss"

METADATA_FILE = f"{VECTOR_DIR}/metadata.pkl"


EMBEDDING_MODEL = "text-embedding-3-small"


def create_embeddings(texts):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return np.array(
        embeddings,
        dtype="float32"
    )


def build_index():

    df = pd.read_csv(
        INPUT_FILE
    )

    texts = df["text"].tolist()

    print(
        f"Creating embeddings for {len(texts)} documents..."
    )

    embeddings = create_embeddings(
        texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    os.makedirs(
        VECTOR_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    metadata = df.to_dict(
        orient="records"
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
        f"FAISS index saved to: {INDEX_FILE}"
    )

    print(
        f"Metadata saved to: {METADATA_FILE}"
    )


if __name__ == "__main__":
    build_index()