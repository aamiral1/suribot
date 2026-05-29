from enum import Enum


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
