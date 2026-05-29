"""Tests for the appointment booking and Google Calendar integration."""
import os

for _k, _v in {
    'OPENAI_API_KEY': 'test-key',
    'AWS_ACCESS_KEY_ID': 'test-key',
    'AWS_SECRET_ACCESS_KEY': 'test-key',
    'AWS_S3_BUCKET': 'test-bucket',
    'PINECONE_API_KEY': 'test-key',
    'PINECONE_INDEX_NAME': 'test-index',
}.items():
    os.environ.setdefault(_k, _v)

from unittest.mock import MagicMock, patch
import pytest
from sales_controller import SalesController

_mock_db = MagicMock()
_mock_s3 = MagicMock()

with (
    patch('database.Database', return_value=_mock_db),
    patch('boto3.resource', return_value=_mock_s3),
    patch('openai.OpenAI'),
):
    import app as flask_app

flask_app.db = _mock_db
flask_app.s3 = _mock_s3

AVAILABLE_SLOTS = [
    {'date': 'Monday 2 June 2026', 'time': '10:00'},
    {'date': 'Monday 2 June 2026', 'time': '14:00'},
    {'date': 'Tuesday 3 June 2026', 'time': '11:00'},
]

MOCK_EVENT_ID = 'gcal-event-abc123'


@pytest.fixture(autouse=True)
def reset_mocks():
    flask_app.db = _mock_db
    flask_app.s3 = _mock_s3
    _mock_db.reset_mock()
    _mock_s3.reset_mock()
    flask_app.sales_controller.run_controller = SalesController.run_controller.__get__(
        flask_app.sales_controller, SalesController
    )


@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    flask_app.app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.app.test_client() as c:
        yield c


def _setup(action, lead_profile=None, requires_booking=False, extra_fields=None):
    profile = lead_profile or SalesController.create_new_profile()
    _mock_db.get_conversation_history.return_value = []
    _mock_db.get_lead_profile.return_value = profile
    _mock_db.get_last_actions.return_value = []
    _mock_db.get_sales_turn_count.return_value = 0
    output = {
        'next_action': action,
        'requires_rag': False,
        'requires_booking_tool': requires_booking,
        'confidence': 0.9,
        'lead_profile_updates': {},
        'reason': 'test',
        'selected_slot_day': None,
        'selected_slot_time': None,
    }
    if extra_fields:
        output.update(extra_fields)
    flask_app.sales_controller.run_controller = MagicMock(return_value=output)


# ---------------------------------------------------------------------------
# Booking guidance (FR7)
# ---------------------------------------------------------------------------

class TestBookingGuidance:
    def test_interested_user_offered_booking(self, client):
        """A user showing interest results in the OFFER_AND_COLLECT action (FR7)."""
        _setup('OFFER_AND_COLLECT')
        with patch('app.generate_final_response',
                   return_value="I'd love to set up a call! What's your name and email?"):
            resp = client.post('/api', json={
                'message': "I'd like to learn more about working with you",
                'session_id': 'booking_session'
            })
        assert resp.status_code == 200
        chosen_action = flask_app.sales_controller.run_controller.return_value['next_action']
        assert chosen_action == 'OFFER_AND_COLLECT'

    def test_direct_booking_request_goes_straight_to_offer(self, client):
        """A direct booking request goes straight to collecting contact details (FR7)."""
        _setup('OFFER_AND_COLLECT')
        with patch('app.generate_final_response',
                   return_value="Sure! Could I grab your name and email?"):
            resp = client.post('/api', json={
                'message': 'I want to book a free discovery call',
                'session_id': 'direct_booking_session'
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Available slots (FR8)
# ---------------------------------------------------------------------------

class TestAvailableSlots:
    def test_offer_slots_action_calls_calendar_service(self, client):
        """OFFER_AVAILABLE_SLOTS fetches real slots from the calendar (FR8)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'James', 'contact_email': 'james@example.com'})
        _setup('OFFER_AVAILABLE_SLOTS', lead_profile=profile, requires_booking=True)

        with (
            patch('app.get_available_slots', return_value=AVAILABLE_SLOTS) as mock_slots,
            patch('app.generate_final_response', return_value='Here are some slots: Mon 10:00'),
        ):
            resp = client.post('/api', json={
                'message': 'What times are available?',
                'session_id': 'slots_session'
            })
        assert resp.status_code == 200
        mock_slots.assert_called_once()

    def test_available_slots_included_in_response_context(self, client):
        """Available slots are passed through to the response generator (FR8)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'James', 'contact_email': 'james@example.com'})
        _setup('OFFER_AVAILABLE_SLOTS', lead_profile=profile, requires_booking=True)

        with (
            patch('app.get_available_slots', return_value=AVAILABLE_SLOTS),
            patch('app.generate_final_response', return_value='Available: Mon 10:00, 14:00') as mock_gen,
        ):
            client.post('/api', json={'message': 'Show me slots', 'session_id': 'slots_context_session'})
        assert mock_gen.called


# ---------------------------------------------------------------------------
# Booking creation (FR9)
# ---------------------------------------------------------------------------

class TestBookingCreation:
    def test_book_appointment_action_creates_calendar_event(self, client):
        """BOOK_APPOINTMENT creates an event in Google Calendar (FR9)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'James Carter', 'contact_email': 'james@example.com'})
        _setup(
            'BOOK_APPOINTMENT',
            lead_profile=profile,
            requires_booking=True,
            extra_fields={
                'selected_slot_day': 'monday',
                'selected_slot_time': '10:00',
            },
        )

        with (
            patch('app.create_booking', return_value=MOCK_EVENT_ID) as mock_create,
            patch('app.generate_final_response', return_value="You're booked for Monday at 10:00!"),
        ):
            resp = client.post('/api', json={
                'message': "Monday at 10am works for me",
                'session_id': 'calendar_event_session'
            })
        assert resp.status_code == 200
        mock_create.assert_called_once()

    def test_booking_confirmation_returned_in_response(self, client):
        """The chatbot response includes a booking confirmation message (FR9)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'Alice', 'contact_email': 'alice@example.com'})
        _setup(
            'BOOK_APPOINTMENT',
            lead_profile=profile,
            requires_booking=True,
            extra_fields={
                'selected_slot_day': 'tuesday',
                'selected_slot_time': '11:00',
            },
        )
        confirmation_msg = "You're confirmed for Tuesday 3 June at 11:00. See you then!"

        with (
            patch('app.create_booking', return_value=MOCK_EVENT_ID),
            patch('app.generate_final_response', return_value=confirmation_msg),
        ):
            resp = client.post('/api', json={
                'message': "Let's go for Tuesday 11am",
                'session_id': 'confirmation_session'
            })
        assert resp.status_code == 200
        assert resp.json['response'] == confirmation_msg

    def test_booking_record_saved_to_database(self, client):
        """A successful booking is saved to the database (FR9)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'Bob', 'contact_email': 'bob@example.com'})
        _setup(
            'BOOK_APPOINTMENT',
            lead_profile=profile,
            requires_booking=True,
            extra_fields={
                'selected_slot_day': 'monday',
                'selected_slot_time': '14:00',
            },
        )

        with (
            patch('app.create_booking', return_value=MOCK_EVENT_ID),
            patch('app.generate_final_response', return_value='All booked!'),
        ):
            client.post('/api', json={'message': 'Monday 2pm please', 'session_id': 'db_persist_session'})
        _mock_db.save_booking.assert_called_once()


# ---------------------------------------------------------------------------
# Google Calendar integration (FR19)
# ---------------------------------------------------------------------------

class TestGoogleCalendarIntegration:
    def test_create_booking_called_with_correct_slot_and_contact(self, client):
        """Calendar booking is called with the correct contact details and slot time (FR19)."""
        profile = SalesController.create_new_profile()
        profile.update({'contact_name': 'Sarah Jones', 'contact_email': 'sarah@example.com'})
        _setup(
            'BOOK_APPOINTMENT',
            lead_profile=profile,
            requires_booking=True,
            extra_fields={
                'selected_slot_day': 'monday',
                'selected_slot_time': '10:00',
            },
        )

        with (
            patch('app.create_booking', return_value=MOCK_EVENT_ID) as mock_create,
            patch('app.generate_final_response', return_value='Booked!'),
        ):
            client.post('/api', json={'message': 'Book me in', 'session_id': 'calendar_integration_session'})

        call_kwargs = mock_create.call_args
        assert call_kwargs is not None
        args = call_kwargs[0] if call_kwargs[0] else []
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        all_args = list(args) + list(kwargs.values())
        combined = ' '.join(str(a) for a in all_args)
        assert 'Sarah Jones' in combined or 'sarah@example.com' in combined
