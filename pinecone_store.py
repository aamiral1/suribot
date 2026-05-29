from pinecone import Pinecone, ServerlessSpec

EMBEDDING_DIMENSION = 1536  # text-embedding-3-small output size
INDEX_METRIC = "dotproduct"  # required for hybrid sparse-dense search
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def get_or_create_index(api_key: str, index_name: str):
    """
    Get or create a sparse-dense Pinecone index with dotproduct metric.

    NOTE: If an existing index uses metric='cosine', it must be deleted manually
    (or via delete_and_recreate_index) before calling this function, as Pinecone
    does not support changing the metric on an existing index.
    """
    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(index_name)


def delete_and_recreate_index(api_key: str, index_name: str):
    """
    Delete the existing index and recreate it as a sparse-dense dotproduct index.
    WARNING: This deletes all stored vectors. Run once during migration only.
    """
    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name in existing:
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=INDEX_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    return pc.Index(index_name)


def query_index(index, embedding: list[float], top_k: int = 3) -> list[dict]:
    """
    Dense-only fallback query (used when BM25 encoder is not yet fitted).
    Backward-compatible with existing free-flow vectors.
    """
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {
            "doc_id": match.metadata["doc_id"],
            "chunk_id": int(match.metadata.get("chunk_id", match.metadata.get("chunk_index", 0))),
            "text": match.metadata["text"],
            "score": match.score,
        }
        for match in results.matches
    ]


def upsert_embeddings(index, doc_id: str, embedded_chunks: list[dict]):
    """Upsert dense-only vectors for a document."""
    vectors = []
    for chunk in embedded_chunks:
        i = chunk["chunk_index"]
        vectors.append({
            "id": f"{doc_id}_{i}",
            "values": chunk["embedding"],
            "metadata": {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk["text"],
            },
        })
    index.upsert(vectors=vectors)
