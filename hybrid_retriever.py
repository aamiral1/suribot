from openai import OpenAI
from pinecone_store import hybrid_query, query_index
import bm25_encoder as bm25_enc


def hybrid_retrieve(
    query: str,
    client: OpenAI,
    pinecone_index,
    top_k: int = 2,
    alpha: float = 0.5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Returns list of dicts:
      {doc_id, chunk_id, text, score}
    """
    dense_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    encoder = bm25_enc.get_encoder()

    if encoder is not None:
        sparse_vector = bm25_enc.encode_query(encoder, query)
        raw_results = hybrid_query(
            pinecone_index,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            alpha=alpha,
        )
    else:
        print("[hybrid_retriever] BM25 encoder not fitted — falling back to dense-only search.")
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
