from chunking_evaluation.chunking import (
    LLMSemanticChunker,
    RecursiveTokenChunker,
    FixedTokenChunker,
)

from chunking_evaluation.utils import openai_token_count


class CustomLLMSemanticChunker(LLMSemanticChunker):
    def __init__(self, organisation="openai", api_key=None, model_name=None, initial_chunk_size=50):
        super().__init__(organisation=organisation, api_key=api_key, model_name=model_name)
        self.splitter = RecursiveTokenChunker(
            chunk_size=initial_chunk_size,
            chunk_overlap=0,
            length_function=openai_token_count
        )


def chunk_text(text: str, api_key: str) -> list[str]:
    chunker = CustomLLMSemanticChunker(
        organisation="openai",
        api_key=api_key,
        model_name="gpt-4.1-mini",
        initial_chunk_size=10,
    )
    return chunker.split_text(text)


def chunk_text_recursive(text: str) -> list[str]:
    chunker = RecursiveTokenChunker(
        chunk_size=300,
        chunk_overlap=0,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return chunker.split_text(text)


def chunk_text_fixed(text: str) -> list[str]:
    chunker = FixedTokenChunker(
        chunk_size=300,
        chunk_overlap=0,
        encoding_name="cl100k_base",
    )
    return chunker.split_text(text)
