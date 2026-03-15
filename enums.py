from enum import Enum


class DocumentStatus(Enum):
    CREATED = "created"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
