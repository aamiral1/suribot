import json
import io
from pinecone_text.sparse import BM25Encoder

BM25_S3_KEY = "bm25_encoder_params.json"

_encoder: BM25Encoder | None = None


def get_encoder() -> BM25Encoder | None:
    """Return the current fitted BM25Encoder, or None if not yet fitted."""
    return _encoder


def fit_and_save(chunk_texts: list[str], s3_client, bucket: str) -> BM25Encoder:
    """
    Fit a BM25Encoder on the given chunks and save params to S3.
    Returns the fitted encoder.
    """
    global _encoder

    encoder = BM25Encoder()
    encoder.fit(chunk_texts)

    # Serialise params to JSON and upload to S3
    params = encoder.get_params()
    params_json = json.dumps(params).encode("utf-8")
    s3_client.meta.client.put_object(
        Bucket=bucket,
        Key=BM25_S3_KEY,
        Body=params_json,
        ContentType="application/json",
    )
    print(f"[bm25_encoder] Fitted on {len(chunk_texts)} chunks and saved to S3: {BM25_S3_KEY}")

    _encoder = encoder
    return encoder


def load_from_s3(s3_client, bucket: str) -> BM25Encoder | None:
    """
    Load a previously fitted BM25Encoder from S3.
    Returns the encoder, or None if no saved params exist.
    """
    global _encoder

    try:
        response = s3_client.meta.client.get_object(Bucket=bucket, Key=BM25_S3_KEY)
        params = json.loads(response["Body"].read().decode("utf-8"))
        encoder = BM25Encoder()
        encoder.set_params(**params)
        print(f"[bm25_encoder] Loaded fitted encoder from S3: {BM25_S3_KEY}")
        _encoder = encoder
        return encoder
    except s3_client.meta.client.exceptions.NoSuchKey:
        print("[bm25_encoder] No saved encoder found in S3.")
        return None
    except Exception as e:
        print(f"[bm25_encoder] Failed to load encoder from S3: {e}")
        return None


def encode_query(encoder: BM25Encoder, query: str) -> dict:
    """Return a Pinecone-compatible sparse vector for a query string."""
    return encoder.encode_queries(query)


def encode_documents(encoder: BM25Encoder, texts: list[str]) -> list[dict]:
    """Return a list of Pinecone-compatible sparse vectors for document texts."""
    return encoder.encode_documents(texts)
