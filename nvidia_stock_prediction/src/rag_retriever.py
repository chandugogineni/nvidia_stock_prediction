import os
import pickle

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


INDEX_FILE = "data/vector_store/nvidia.faiss"

METADATA_FILE = "data/vector_store/metadata.pkl"

EMBEDDING_MODEL = "text-embedding-3-small"


def load_vector_store():

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return index, metadata


def embed_query(query):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    return np.array(
        [response.data[0].embedding],
        dtype="float32"
    )


def retrieve_news(
    query,
    top_k=5
):

    index, metadata = load_vector_store()

    query_embedding = embed_query(
        query
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < len(metadata):

            document = metadata[idx]

            results.append({
                "title": document["title"],
                "published_at": document["published_at"],
                "text": document["text"],
                "url": document["url"],
                "distance": float(distance)
            })

    return results


if __name__ == "__main__":

    results = retrieve_news(
        "NVIDIA AI GPU demand and earnings outlook",
        top_k=5
    )

    for result in results:

        print("\n" + "=" * 70)

        print(
            result["title"]
        )

        print(
            result["published_at"]
        )

        print(
            result["text"]
        )