import os
import json
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import custom_exceptions as ex

SCOPES = ["https://www.googleapis.com/auth/calendar"]
LONDON = ZoneInfo("Europe/London")
SLOT_DURATION = 15  # minutes
BUSINESS_START = time(9, 0)
BUSINESS_END = time(17, 0)


def _get_service():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "google_token.json")

    if not os.path.exists(token_path):
        raise ex.CalendarAuthError(
            f"Google Calendar token not found at '{token_path}'. "
            "Run scripts/google_auth.py to authorise access."
        )

    try:
        with open(token_path) as f:
            data = json.load(f)

        creds = Credentials.from_authorized_user_info(data, SCOPES)

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
            else:
                raise ex.CalendarAuthError(
                    "Google Calendar credentials are invalid. Re-run scripts/google_auth.py."
                )

        return build("calendar", "v3", credentials=creds)

    except ex.CalendarAuthError:
        raise
    except Exception as e:
        raise ex.CalendarAuthError(f"Failed to load Google Calendar credentials: {e}") from e


def _is_slot_free(service, calendar_id: str, slot_dt: datetime, slot_end: datetime) -> bool:
    body = {
        "timeMin": slot_dt.isoformat(),
        "timeMax": slot_end.isoformat(),
        "items": [{"id": calendar_id}],
    }
    result = service.freebusy().query(body=body).execute()
    busy = result["calendars"][calendar_id].get("busy", [])
    return len(busy) == 0


def get_available_slots(start_dt: datetime, end_dt: datetime) -> list[str]:
    """Return up to 3 free 15-min slot start times as ISO 8601 strings (Europe/London)."""
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")

    try:
        service = _get_service()

        start_dt = start_dt.astimezone(LONDON)
        end_dt = end_dt.astimezone(LONDON)

        body = {
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "items": [{"id": calendar_id}],
        }
        result = service.freebusy().query(body=body).execute()
        busy_raw = result["calendars"][calendar_id].get("busy", [])

        busy_intervals = []
        for interval in busy_raw:
            b_start = datetime.fromisoformat(
                interval["start"].replace("Z", "+00:00")
            ).astimezone(LONDON)
            b_end = datetime.fromisoformat(
                interval["end"].replace("Z", "+00:00")
            ).astimezone(LONDON)
            busy_intervals.append((b_start, b_end))

        # Align start to next 15-min boundary
        now = datetime.now(LONDON)
        slot = start_dt.replace(second=0, microsecond=0)
        remainder = slot.minute % SLOT_DURATION
        if remainder != 0:
            slot += timedelta(minutes=SLOT_DURATION - remainder)
        slot = slot.replace(second=0, microsecond=0)

        delta = timedelta(minutes=SLOT_DURATION)
        free_slots = []

        while slot < end_dt and len(free_slots) < 3:
            if slot <= now:
                slot += delta
                continue
            if slot.weekday() < 5:
                slot_time = slot.time()
                if BUSINESS_START <= slot_time < BUSINESS_END:
                    slot_end = slot + delta
                    if slot_end.time() <= BUSINESS_END:
                        is_busy = any(
                            slot < b_end and slot_end > b_start
                            for b_start, b_end in busy_intervals
                        )
                        if not is_busy:
                            free_slots.append(slot.isoformat())
            slot += delta

        return free_slots

    except ex.CalendarAuthError:
        raise
    except Exception as e:
        raise ex.CalendarServiceError(f"Failed to fetch available slots: {e}") from e


def create_booking(slot_iso: str, name: str, email: str) -> str:
    """Create a 15-min discovery call event. Returns the Google Calendar event_id."""
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")

    try:
        slot_dt = datetime.fromisoformat(slot_iso).astimezone(LONDON)
        slot_end = slot_dt + timedelta(minutes=SLOT_DURATION)

        service = _get_service()

        if not _is_slot_free(service, calendar_id, slot_dt, slot_end):
            raise ex.SlotAlreadyTakenError(f"Slot {slot_iso} is no longer available.")

        event = {
            "summary": f"Discovery Call - {name}",
            "description": "Discovery call booked via Suri Marketing chatbot.",
            "start": {"dateTime": slot_dt.isoformat(), "timeZone": "Europe/London"},
            "end": {"dateTime": slot_end.isoformat(), "timeZone": "Europe/London"},
        }

        if email:
            event["attendees"] = [{"email": email}]

        created = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates="all" if email else "none",
        ).execute()

        return created["id"]

    except (ex.CalendarAuthError, ex.SlotAlreadyTakenError):
        raise
    except Exception as e:
        raise ex.CalendarServiceError(f"Failed to create booking: {e}") from e


def reschedule_booking(event_id: str, new_slot_iso: str) -> None:
    """Update the start/end time of an existing calendar event."""
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")

    try:
        new_slot_dt = datetime.fromisoformat(new_slot_iso).astimezone(LONDON)
        new_slot_end = new_slot_dt + timedelta(minutes=SLOT_DURATION)

        service = _get_service()

        if not _is_slot_free(service, calendar_id, new_slot_dt, new_slot_end):
            raise ex.SlotAlreadyTakenError(f"Slot {new_slot_iso} is not available.")

        try:
            event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        except Exception:
            raise ex.EventNotFoundError(f"Event '{event_id}' not found on calendar.")

        event["start"] = {"dateTime": new_slot_dt.isoformat(), "timeZone": "Europe/London"}
        event["end"] = {"dateTime": new_slot_end.isoformat(), "timeZone": "Europe/London"}

        service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event,
            sendUpdates="all",
        ).execute()

    except (ex.CalendarAuthError, ex.SlotAlreadyTakenError, ex.EventNotFoundError):
        raise
    except Exception as e:
        raise ex.CalendarServiceError(f"Failed to reschedule booking: {e}") from e
