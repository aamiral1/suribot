"""Tests for the analytics monitoring and admin auth endpoints."""
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

SAMPLE_SUMMARY = {
    'total_sessions': 42,
    'total_messages': 187,
    'total_bookings': 9,
    'daily': [
        {'date': '2026-05-21', 'conversations': 4},
        {'date': '2026-05-22', 'conversations': 7},
    ],
}

SAMPLE_BOOKINGS = [
    {
        'session_id': 'sess-001',
        'name': 'James Carter',
        'email': 'james@example.com',
        'slot_iso': '2026-06-02T10:00:00+01:00',
        'status': 'confirmed',
    },
    {
        'session_id': 'sess-002',
        'name': 'Alice Patel',
        'email': 'alice@example.com',
        'slot_iso': '2026-06-03T14:00:00+01:00',
        'status': 'confirmed',
    },
]


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


@pytest.fixture
def logged_in_client(client):
    with client.session_transaction() as sess:
        sess['logged_in'] = True
    return client


# ---------------------------------------------------------------------------
# Admin auth protection (FR16)
# ---------------------------------------------------------------------------

class TestAdminAuthProtection:
    def test_analytics_page_redirects_without_login(self, client):
        """Unauthenticated requests to /analytics are redirected to login (FR16)."""
        resp = client.get('/analytics', follow_redirects=False)
        assert resp.status_code == 302
        assert b'login' in resp.headers['Location'].lower().encode()

    def test_analytics_summary_blocked_without_login(self, client):
        """Analytics summary endpoint is protected by login (FR16)."""
        resp = client.get('/analytics/summary', follow_redirects=False)
        assert resp.status_code == 302

    def test_analytics_bookings_blocked_without_login(self, client):
        """Analytics bookings endpoint is protected by login (FR16)."""
        resp = client.get('/analytics/bookings', follow_redirects=False)
        assert resp.status_code == 302

    def test_login_page_is_publicly_accessible(self, client):
        """The login page is publicly accessible without auth (FR16)."""
        resp = client.get('/login')
        assert resp.status_code == 200

    def test_invalid_credentials_rejected_with_401(self, client):
        """Wrong credentials are rejected with 401 (FR16)."""
        resp = client.post('/login', json={
            'username': 'admin',
            'password': 'wrongpassword'
        })
        assert resp.status_code == 401

    def test_admin_dashboard_blocked_without_login(self, client):
        """Admin dashboard redirects when not logged in (FR16)."""
        resp = client.get('/admin', follow_redirects=False)
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Analytics dashboard (FR14)
# ---------------------------------------------------------------------------

class TestAnalyticsDashboard:
    def test_analytics_page_loads_when_logged_in(self, logged_in_client):
        """Analytics page loads correctly for a logged-in admin (FR14)."""
        resp = logged_in_client.get('/analytics')
        assert resp.status_code == 200

    def test_analytics_summary_returns_json_for_7_days(self, logged_in_client):
        """Summary endpoint returns the expected JSON shape for a 7-day window (FR14)."""
        _mock_db.get_analytics_summary.return_value = SAMPLE_SUMMARY
        resp = logged_in_client.get('/analytics/summary?days=7')
        assert resp.status_code == 200
        data = resp.json
        assert 'total_sessions' in data
        assert 'total_messages' in data
        assert 'daily' in data

    def test_analytics_summary_returns_correct_counts(self, logged_in_client):
        """Summary values match what the database returned (FR14)."""
        _mock_db.get_analytics_summary.return_value = SAMPLE_SUMMARY
        resp = logged_in_client.get('/analytics/summary?days=7')
        assert resp.json['total_sessions'] == 42
        assert resp.json['total_messages'] == 187

    def test_analytics_summary_accepts_30_day_window(self, logged_in_client):
        """Summary endpoint passes the 30-day window to the database (FR14)."""
        _mock_db.get_analytics_summary.return_value = SAMPLE_SUMMARY
        resp = logged_in_client.get('/analytics/summary?days=30')
        assert resp.status_code == 200
        _mock_db.get_analytics_summary.assert_called_with(30)


# ---------------------------------------------------------------------------
# Booking records display (FR15)
# ---------------------------------------------------------------------------

class TestBookingRecords:
    def test_bookings_endpoint_returns_list(self, logged_in_client):
        """Bookings endpoint returns a JSON list (FR15)."""
        _mock_db.get_bookings_for_period.return_value = SAMPLE_BOOKINGS
        resp = logged_in_client.get('/analytics/bookings')
        assert resp.status_code == 200
        assert isinstance(resp.json, list)

    def test_bookings_list_contains_expected_records(self, logged_in_client):
        """Booking records contain the expected name and email fields (FR15)."""
        _mock_db.get_bookings_for_period.return_value = SAMPLE_BOOKINGS
        resp = logged_in_client.get('/analytics/bookings')
        bookings = resp.json
        assert len(bookings) == 2
        assert bookings[0]['name'] == 'James Carter'
        assert bookings[1]['email'] == 'alice@example.com'

    def test_bookings_accepts_custom_day_window(self, logged_in_client):
        """A custom day window is forwarded to the database query (FR15)."""
        _mock_db.get_bookings_for_period.return_value = SAMPLE_BOOKINGS
        resp = logged_in_client.get('/analytics/bookings?days=30')
        assert resp.status_code == 200
        _mock_db.get_bookings_for_period.assert_called_with(30)


# ---------------------------------------------------------------------------
# Record persistence (FR18)
# ---------------------------------------------------------------------------

class TestRecordPersistence:
    def test_conversation_message_saved_after_chat(self, client):
        """Each chat turn is persisted to the conversation history (FR18)."""
        _mock_db.get_conversation_history.return_value = []
        _mock_db.get_lead_profile.return_value = SalesController.create_new_profile()
        _mock_db.get_last_actions.return_value = []
        _mock_db.get_sales_turn_count.return_value = 0
        flask_app.sales_controller.run_controller = MagicMock(return_value={
            'next_action': 'ANSWER_FROM_CONTEXT',
            'requires_rag': False,
            'requires_booking_tool': False,
            'confidence': 0.9,
            'lead_profile_updates': {},
            'reason': 'test',
        })
        with patch('app.generate_final_response', return_value='Here to help!'):
            client.post('/api', json={
                'message': 'Tell me about Suri Marketing',
                'session_id': 'analytics_session'
            })
        _mock_db.save_message.assert_called()

    def test_session_created_on_first_message(self, client):
        """A new session record is created on first use of a session_id (FR18)."""
        _mock_db.get_conversation_history.return_value = []
        _mock_db.get_lead_profile.return_value = SalesController.create_new_profile()
        _mock_db.get_last_actions.return_value = []
        _mock_db.get_sales_turn_count.return_value = 0
        flask_app.sales_controller.run_controller = MagicMock(return_value={
            'next_action': 'ANSWER_FROM_CONTEXT',
            'requires_rag': False,
            'requires_booking_tool': False,
            'confidence': 0.9,
            'lead_profile_updates': {},
            'reason': 'test',
        })
        with patch('app.generate_final_response', return_value='Hello!'):
            client.post('/api', json={
                'message': 'Hello',
                'session_id': 'fresh_session'
            })
        _mock_db.ensure_session.assert_called_with('fresh_session')
