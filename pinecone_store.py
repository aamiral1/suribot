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
        print(f"[pinecone] Created new hybrid index: {index_name}")
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
        print(f"[pinecone] Deleted existing index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=INDEX_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print(f"[pinecone] Recreated hybrid index: {index_name}")
    return pc.Index(index_name)


def upsert_chunks(index, doc_id: str, embedded_chunks: list[dict], sparse_vectors: list[dict]):
    """
    Upsert chunks with both dense and sparse vectors.

    embedded_chunks: list of {chunk_id, text, embedding}
    sparse_vectors:  list of Pinecone sparse dicts {indices: [...], values: [...]}
                     in the same order as embedded_chunks
    """
    vectors = []
    for chunk, sparse in zip(embedded_chunks, sparse_vectors):
        vector_id = f"{doc_id}_p{chunk['chunk_id']}"
        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "sparse_values": sparse,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            },
        })
    index.upsert(vectors=vectors)
    print(f"[pinecone] Upserted {len(vectors)} hybrid vectors for doc {doc_id}")


def hybrid_query(index, dense_vector: list[float], sparse_vector: dict, top_k: int, alpha: float = 0.5) -> list[dict]:
    """
    Query the hybrid index with both dense and sparse vectors.

    alpha=1.0 → pure semantic (dense only)
    alpha=0.0 → pure keyword (sparse only)
    alpha=0.5 → equal blend (default)

    Vectors are scaled by alpha and (1-alpha) before querying so Pinecone's
    dotproduct scoring produces the weighted blend.
    """
    # Scale dense vector
    scaled_dense = [v * alpha for v in dense_vector]

    # Scale sparse vector
    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values": [v * (1 - alpha) for v in sparse_vector["values"]],
    }

    results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "doc_id": match.metadata["doc_id"],
            "chunk_id": int(match.metadata["chunk_id"]),
            "text": match.metadata["text"],
            "score": match.score,
        }
        for match in results.matches
    ]


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
    """
    Legacy dense-only upsert — kept for reference but superseded by upsert_chunks.
    """
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
    print(f"[Pinecone] Upserted {len(vectors)} vectors for doc {doc_id}")
