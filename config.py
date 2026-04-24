import os
from enum import Enum


class ChunkingStrategy(str, Enum):
    SECTION_AND_SEMANTIC = "section_and_semantic"
    SEMANTIC_ONLY = "semantic_only"
    RECURSIVE_TOKEN = "recursive_token"
    FIXED_SIZE = "fixed_size"


def get_chunking_strategy() -> ChunkingStrategy:
    raw = os.getenv("CHUNKING_STRATEGY", "section_and_semantic").strip().lower()
    try:
        return ChunkingStrategy(raw)
    except ValueError:
        valid = [s.value for s in ChunkingStrategy]
        raise ValueError(
            f"Invalid CHUNKING_STRATEGY='{raw}'. Must be one of: {valid}"
        )


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"   # alpha = 0.5
    DENSE  = "dense"    # alpha = 1.0
    SPARSE = "sparse"   # alpha = 0.0


def get_retrieval_alpha() -> float:
    raw = os.getenv("RETRIEVAL_STRATEGY", "hybrid").strip().lower()
    return {"hybrid": 0.5, "dense": 1.0, "sparse": 0.0}.get(raw, 0.5)
