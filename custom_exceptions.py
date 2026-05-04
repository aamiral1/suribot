class ExtractionTimeOut(Exception):
    """Exception raised when OpenAI calls timeout"""

    pass


class InvalidDocumentStatusTransition(Exception):
    """Exception raised when requested state change is invalid"""

    pass


class InvalidDocumentID(Exception):
    """Exception raised when searching for doc id that is not present in DB"""

    pass


class InvalidFileType(Exception):
    """Exception raised when a file has an unsupported or missing extension"""

    pass


class CalendarAuthError(Exception):
    """Raised when the Google Calendar OAuth2 token is missing or credentials are invalid."""

    pass


class CalendarServiceError(Exception):
    """Raised when the Google Calendar API returns an unexpected error."""

    pass


class SlotAlreadyTakenError(Exception):
    """Raised when a requested calendar slot is no longer free at booking time."""

    pass


class EventNotFoundError(Exception):
    """Raised when the given Google Calendar event_id does not exist."""

    pass
