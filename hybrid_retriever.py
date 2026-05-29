from openai import OpenAI
from pinecone_store import query_index


def hybrid_retrieve(
    query: str,
    client: OpenAI,
    pinecone_index,
    top_k: int = 2,
) -> list[dict]:
    dense_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    raw_results = query_index(pinecone_index, dense_vector, top_k=top_k)

    return [
        {
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "score": r["score"],
        }
        for r in raw_results
    ]
