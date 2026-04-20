from openai import OpenAI


def embed_chunks(client: OpenAI, chunks: list[str]) -> list[dict]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    return [
        {
            "chunk_index": i,
            "text": chunks[i],
            "embedding": item.embedding,
        }
        for i, item in enumerate(response.data)
    ]
