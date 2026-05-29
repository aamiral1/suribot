from chunking_evaluation.chunking import FixedTokenChunker


def chunk_text_fixed(text: str) -> list[str]:
    chunker = FixedTokenChunker(
        chunk_size=300,
        chunk_overlap=0,
        encoding_name="cl100k_base",
    )
    return chunker.split_text(text)
