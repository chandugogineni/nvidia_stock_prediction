import pickle

import faiss

from sentence_transformers import (
    SentenceTransformer
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


class NVIDIARetriever:

    def __init__(self):

        print(
            "Loading FAISS..."
        )

        self.index = (
            faiss.read_index(
                INDEX_FILE
            )
        )

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            self.metadata = (
                pickle.load(f)
            )

        self.embedding_model = (
            SentenceTransformer(
                MODEL_NAME
            )
        )

    def search(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            document = (
                self.metadata[index]
            )

            results.append(
                {
                    "score": float(score),
                    "title": document.get(
                        "title"
                    ),
                    "published_at": document.get(
                        "published_at"
                    ),
                    "url": document.get(
                        "url"
                    ),
                    "text": document.get(
                        "text"
                    )
                }
            )

        return results


if __name__ == "__main__":

    retriever = (
        NVIDIARetriever()
    )

    results = retriever.search(
        "NVIDIA AI GPU demand",
        top_k=5
    )

    for i, result in enumerate(
        results,
        1
    ):

        print("\n" + "=" * 60)

        print(
            f"Result {i}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Date: "
            f"{result['published_at']}"
        )

        print(
            result["text"]
        )