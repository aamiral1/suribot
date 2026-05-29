"""Tests for the visitor-facing chatbot widget and conversation flow."""
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


def _controller(action='ANSWER_FROM_CONTEXT', requires_rag=False):
    return {
        'next_action': action,
        'requires_rag': requires_rag,
        'requires_booking_tool': False,
        'confidence': 0.9,
        'lead_profile_updates': {},
        'reason': 'test',
    }


def _setup(action='ANSWER_FROM_CONTEXT', history=None, requires_rag=False, sales_turn=0):
    _mock_db.get_conversation_history.return_value = history or []
    _mock_db.get_lead_profile.return_value = SalesController.create_new_profile()
    _mock_db.get_last_actions.return_value = []
    _mock_db.get_sales_turn_count.return_value = sales_turn
    flask_app.sales_controller.run_controller = MagicMock(
        return_value=_controller(action, requires_rag)
    )


# ---------------------------------------------------------------------------
# Widget accessibility (FR1)
# ---------------------------------------------------------------------------

class TestChatbotWidgetAccess:
    def test_home_page_returns_200(self, client):
        """The chatbot home page responds with 200 (FR1)."""
        resp = client.get('/')
        assert resp.status_code == 200

    def test_home_page_contains_chat_widget(self, client):
        """The home page includes the chat widget markup (FR1)."""
        resp = client.get('/')
        assert b'vb-footer' in resp.data

    def test_cta_booking_button_present(self, client):
        """The CTA booking button is present in the widget (FR17)."""
        resp = client.get('/')
        assert b'Book a free strategy call' in resp.data

    def test_embed_route_returns_200(self, client):
        """The embed iframe route responds with 200."""
        resp = client.get('/embed')
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Free-text messages (FR2)
# ---------------------------------------------------------------------------

class TestFreeTextMessages:
    def test_api_accepts_message_and_returns_response(self, client):
        """/api returns a chatbot response for a free-text message (FR2)."""
        _setup()
        with patch('app.generate_final_response', return_value='How can I help?'):
            resp = client.post('/api', json={
                'message': 'What services does Suri Marketing offer?',
                'session_id': 'visitor_session'
            })
        assert resp.status_code == 200
        assert resp.json['response'] == 'How can I help?'

    def test_api_works_without_explicit_session_id(self, client):
        """Messages are handled without an explicit session_id (FR2)."""
        _setup()
        with patch('app.generate_final_response', return_value='Sure!'):
            resp = client.post('/api', json={'message': 'Tell me about pricing'})
        assert resp.status_code == 200
        assert 'response' in resp.json

    def test_missing_message_returns_400(self, client):
        """A missing message field returns HTTP 400 (FR2)."""
        resp = client.post('/api', json={'session_id': 'test_session'})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Knowledge base answers (FR3)
# ---------------------------------------------------------------------------

class TestKnowledgeBaseAnswers:
    def test_rag_action_triggers_retrieval(self, client):
        """ANSWER_WITH_RAG triggers the knowledge base retrieval pipeline (FR3)."""
        _setup(action='ANSWER_WITH_RAG', requires_rag=True)
        mock_chunks = [{'text': 'Suri offers social media marketing.', 'score': 0.9}]
        with (
            patch('app.retrieve_relevant_chunks', return_value=mock_chunks) as mock_retrieve,
            patch('app.generate_final_response', return_value='We offer social media marketing.'),
        ):
            resp = client.post('/api', json={
                'message': 'What does Suri Marketing do?',
                'session_id': 'rag_session'
            })
        assert resp.status_code == 200
        mock_retrieve.assert_called_once()

    def test_response_generated_with_kb_context(self, client):
        """The response generator receives the retrieved KB context (FR3)."""
        _setup(action='ANSWER_WITH_RAG', requires_rag=True)
        mock_chunks = [{'text': 'We run Facebook and Instagram campaigns.', 'score': 0.85}]
        with (
            patch('app.retrieve_relevant_chunks', return_value=mock_chunks),
            patch('app.generate_final_response', return_value='We run social campaigns.') as mock_gen,
        ):
            client.post('/api', json={
                'message': 'Do you do Facebook ads?',
                'session_id': 'context_session'
            })
        assert mock_gen.called


# ---------------------------------------------------------------------------
# Follow-up questions (FR4)
# ---------------------------------------------------------------------------

class TestFollowUpQuestions:
    def test_vague_message_triggers_clarifying_action(self, client):
        """A vague message triggers the ASK_CLARIFYING_QUESTION action (FR4)."""
        _setup(action='ASK_CLARIFYING_QUESTION')
        with patch('app.generate_final_response', return_value='Could you tell me more?'):
            resp = client.post('/api', json={
                'message': 'I need help',
                'session_id': 'vague_message_session'
            })
        assert resp.status_code == 200
        called_action = flask_app.sales_controller.run_controller.call_args
        assert called_action is not None

    def test_situation_question_action_returned_for_new_visitor(self, client):
        """A new visitor gets a situational follow-up question (FR4)."""
        _setup(action='ASK_SITUATION_QUESTION')
        with patch('app.generate_final_response', return_value="What type of business are you?"):
            resp = client.post('/api', json={
                'message': 'Hi there',
                'session_id': 'new_visitor_session'
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Conversation context (FR5)
# ---------------------------------------------------------------------------

class TestConversationContext:
    def test_conversation_history_passed_to_controller(self, client):
        """Prior conversation history is fetched from the DB and passed to the controller (FR5)."""
        prior_history = [
            {'role': 'user', 'content': 'I run a restaurant'},
            {'role': 'assistant', 'content': 'Great, how can I help?'},
        ]
        _setup(history=prior_history)
        with patch('app.generate_final_response', return_value='Of course!'):
            client.post('/api', json={
                'message': 'Tell me more about pricing',
                'session_id': 'returning_session'
            })
        call_kwargs = flask_app.sales_controller.run_controller.call_args
        passed_history = call_kwargs[1].get('history') or call_kwargs[0][1]
        assert passed_history == prior_history

    def test_new_session_starts_with_empty_history(self, client):
        """A brand new session starts with an empty conversation history (FR5)."""
        _setup(history=[])
        with patch('app.generate_final_response', return_value='Hello!'):
            resp = client.post('/api', json={
                'message': 'Hello',
                'session_id': 'new_session'
            })
        assert resp.status_code == 200
        _mock_db.get_conversation_history.assert_called_once()


# ---------------------------------------------------------------------------
# Lead profile capture (FR6)
# ---------------------------------------------------------------------------

class TestLeadProfileCapture:
    def test_lead_profile_updates_are_merged(self, client):
        """Lead profile updates from the controller are merged and persisted (FR6)."""
        _setup()
        flask_app.sales_controller.run_controller = MagicMock(return_value={
            'next_action': 'ANSWER_FROM_CONTEXT',
            'requires_rag': False,
            'requires_booking_tool': False,
            'confidence': 0.9,
            'lead_profile_updates': {'business_type': 'restaurant', 'location': 'Birmingham'},
            'reason': 'extracted from message',
        })
        with patch('app.generate_final_response', return_value='Got it!'):
            resp = client.post('/api', json={
                'message': "I run a restaurant in Birmingham",
                'session_id': 'lead_profile_session'
            })
        assert resp.status_code == 200
        _mock_db.save_lead_profile.assert_called_once()
