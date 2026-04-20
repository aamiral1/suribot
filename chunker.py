# Main Chunking Functions
from chunking_evaluation.chunking import (
    LLMSemanticChunker,
    RecursiveTokenChunker,
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
