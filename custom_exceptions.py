class ExtractionTimeOut(Exception):
    """Exception raised when OpenAI calls timeout"""

    pass


class InvalidDocumentStatusTransition(Exception):
    """Exception raised when requested state change is invalid"""

    pass


class InvalidDocumentID(Exception):
    """Exception raised when searching for doc id that is not present in DB"""

    pass
