from enum import Enum
import exceptions as ex


class DocumentStatus(Enum):
    CREATED = "created"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"

class SourceType(Enum):
    UPLOAD = "upload"
    CRAWL = "crawl"

class AllowedFileTypes(Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    MD = "MD"
    PNG = "PNG"

    @classmethod
    def from_filename(cls, filename: str):
        import os

        if not filename or "." not in filename:
            raise ex.InvalidFileType("Invalid file name")

        ext = os.path.splitext(filename)[1][1:].upper()

        try:
            return cls(ext)
        except ValueError:
            raise ex.InvalidFileType(f"Unsupported file type: {ext}")