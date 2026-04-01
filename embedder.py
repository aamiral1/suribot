from openai import OpenAI


def embed_chunks(client: OpenAI, chunks: list[str]) -> list[dict]:
    results = []

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk,
        )
        embedding = response.data[0].embedding

        print(f"\n--- CHUNK {i + 1} / {len(chunks)} ---")
        print(f"Text: {chunk}")
        print(f"Embedding (first 5 dims): {embedding[:5]}")

        results.append({
            "chunk_index": i,
            "text": chunk,
            "embedding": embedding,
        })

    return results
