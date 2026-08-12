from rag_retriever import (
    NVIDIARetriever
)

from ollama_llm import (
    analyze_news
)


def main():

    retriever = (
        NVIDIARetriever()
    )

    query = """
    NVIDIA AI GPU demand,
    data center growth,
    earnings outlook,
    competition and regulatory risks
    """

    results = retriever.search(
        query,
        top_k=5
    )

    print(
        f"Retrieved {len(results)} articles"
    )

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
TITLE:
{result['title']}

DATE:
{result['published_at']}

CONTENT:
{result['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    print(
        "\nSending context to Ollama..."
    )

    features = analyze_news(
        context
    )

    print(
        "\nLLM FEATURES"
    )

    print(
        features
    )


if __name__ == "__main__":
    main()