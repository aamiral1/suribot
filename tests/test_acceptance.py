import os

# Set env vars BEFORE importing app — module-level globals read these on startup
for _k, _v in {
    'OPENAI_API_KEY': 'test-key',
    'AWS_ACCESS_KEY_ID': 'test-key',
    'AWS_SECRET_ACCESS_KEY': 'test-key',
    'AWS_S3_BUCKET': 'test-bucket',
    'PINECONE_API_KEY': 'test-key',
    'PINECONE_INDEX_NAME': 'test-index',
}.items():
    os.environ.setdefault(_k, _v)

from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import botocore.exceptions

from enums import DocumentStatus
from sales_controller import SalesController
import custom_exceptions as ex

# ---------------------------------------------------------------------------
# Patch heavy dependencies before importing app so module-level init doesn't
# hit real services (Postgres, S3, OpenAI, BM25/S3).
# ---------------------------------------------------------------------------
_mock_db = MagicMock()
_mock_s3 = MagicMock()

with (
    patch('database.Database', return_value=_mock_db),
    patch('boto3.resource', return_value=_mock_s3),
    patch('openai.OpenAI'),
    patch('bm25_encoder.load_from_s3', return_value=None),
    patch('bm25_encoder.fit_and_save'),
):
    import app as flask_app

# Replace globals so per-test mock configuration takes effect
flask_app.db = _mock_db
flask_app.s3 = _mock_s3


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_db.reset_mock()
    _mock_s3.reset_mock()
    # restore sales_controller.run_controller to the real method after each test
    flask_app.sales_controller.run_controller = SalesController.run_controller.__get__(
        flask_app.sales_controller, SalesController
    )


@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    flask_app.app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _controller_output(action='ANSWER_FROM_CONTEXT', requires_rag=False, requires_booking=False):
    return {
        'next_action': action,
        'requires_rag': requires_rag,
        'requires_booking_tool': requires_booking,
        'confidence': 0.9,
        'lead_profile_updates': {},
        'reason': 'test',
    }


def _setup_chatbot_mocks(sales_turn=0, history=None, action='ANSWER_FROM_CONTEXT',
                          requires_rag=False, requires_booking=False):
    _mock_db.get_conversation_history.return_value = history or []
    _mock_db.get_lead_profile.return_value = SalesController.create_new_profile()
    _mock_db.get_last_actions.return_value = []
    _mock_db.get_sales_turn_count.return_value = sales_turn
    flask_app.sales_controller.run_controller = MagicMock(
        return_value=_controller_output(action, requires_rag, requires_booking)
    )


class _SyncThread:
    """Replaces threading.Thread — runs target synchronously so we can assert on results."""
    def __init__(self, target=None, args=(), **kwargs):
        self._target = target
        self._args = args

    def start(self):
        if self._target:
            self._target(*self._args)


# ---------------------------------------------------------------------------
# Phase 1 — Chatbot API basic
# ---------------------------------------------------------------------------

class TestChatbotAPIBasic:
    def test_valid_message_and_session_returns_response(self, client):
        _setup_chatbot_mocks()
        with patch('app.generate_final_response', return_value='Here to help!'):
            resp = client.post('/api', json={'message': 'hello', 'session_id': 'sess-1'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert 'response' in body
        assert isinstance(body['response'], str)

    def test_valid_message_without_session_id_still_works(self, client):
        _setup_chatbot_mocks()
        with patch('app.generate_final_response', return_value='Hello!'):
            resp = client.post('/api', json={'message': 'hi'})
        assert resp.status_code == 200
        assert 'response' in resp.get_json()

    def test_missing_message_returns_400(self, client):
        resp = client.post('/api', json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Phase 2 — Admin: document management
# ---------------------------------------------------------------------------

class TestDocumentList:
    def test_documents_returned_as_list(self, client):
        _mock_db.get_all_documents.return_value = [
            ('doc-1', 'upload', 'file.pdf', '10 KB', 'PDF',
             '2024-01-01', None, None, 'created', None, None, None, False)
        ]
        resp = client.get('/documents')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'documents' in data
        assert len(data['documents']) == 1
        assert data['documents'][0]['doc_id'] == 'doc-1'

    def test_db_error_returns_500(self, client):
        _mock_db.get_all_documents.side_effect = Exception('DB down')
        resp = client.get('/documents')
        assert resp.status_code == 500
        assert 'error' in resp.get_json()
        _mock_db.get_all_documents.side_effect = None


class TestDocumentStatus:
    def test_processing_state(self, client):
        _mock_db.get_status.return_value = DocumentStatus.PROCESSING
        resp = client.get('/document/doc-1/status')
        data = resp.get_json()
        assert data['status'] == 'processing'
        assert data['has_text'] is False

    def test_created_state(self, client):
        _mock_db.get_status.return_value = DocumentStatus.CREATED
        resp = client.get('/document/doc-1/status')
        data = resp.get_json()
        assert data['status'] == 'created'
        assert data['has_text'] is False

    def test_failed_state(self, client):
        _mock_db.get_status.return_value = DocumentStatus.FAILED
        resp = client.get('/document/doc-1/status')
        data = resp.get_json()
        assert data['status'] == 'failed'
        assert data['has_text'] is False

    def test_success_with_file_in_s3(self, client):
        _mock_db.get_status.return_value = DocumentStatus.SUCCESS
        _mock_db.get_extracted_text_file_path.return_value = ('test-bucket', 'doc-1.txt')
        _mock_s3.meta.client.head_object.return_value = {}
        resp = client.get('/document/doc-1/status')
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['has_text'] is True

    def test_success_with_file_missing_from_s3(self, client):
        _mock_db.get_status.return_value = DocumentStatus.SUCCESS
        _mock_db.get_extracted_text_file_path.return_value = ('test-bucket', 'doc-1.txt')
        _mock_s3.meta.client.head_object.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': '404'}}, 'HeadObject'
        )
        resp = client.get('/document/doc-1/status')
        data = resp.get_json()
        assert data['status'] == 'error'
        assert data['has_text'] is False
        _mock_s3.meta.client.head_object.side_effect = None


class TestExtractText:
    def test_created_doc_begins_processing(self, client):
        _mock_db.get_status.return_value = DocumentStatus.CREATED
        _mock_db.get_file_path.return_value = ('test-bucket', 'file.pdf')
        with patch('threading.Thread', _SyncThread):
            with patch('app.extract_doc_info', return_value='Extracted text.'):
                resp = client.post('/extract-text', json={'doc_id': 'doc-1'})
        assert resp.status_code == 200
        assert resp.get_json()['status'] == 'began processing'
        _mock_db.transition_status.assert_any_call(
            doc_id='doc-1', new_status=DocumentStatus.PROCESSING
        )

    def test_already_processing_doc_not_re_triggered(self, client):
        _mock_db.get_status.return_value = DocumentStatus.PROCESSING
        resp = client.post('/extract-text', json={'doc_id': 'doc-1'})
        assert resp.status_code == 200
        assert resp.get_json()['status'] == 'already processing'

    def test_already_extracted_doc_is_idempotent(self, client):
        _mock_db.get_status.return_value = DocumentStatus.SUCCESS
        resp = client.post('/extract-text', json={'doc_id': 'doc-1'})
        assert resp.status_code == 200
        assert resp.get_json()['status'] == 'already extracted'


class TestKBStatus:
    def test_kb_status_endpoint_returns_status(self, client):
        _mock_db.get_kb_status.return_value = 'processing'
        resp = client.get('/document/doc-1/kb-status')
        assert resp.status_code == 200
        assert resp.get_json()['kb_status'] == 'processing'


# ---------------------------------------------------------------------------
# Phase 3 — Website parsing pipeline
# ---------------------------------------------------------------------------

_EMBEDDED = [{'chunk_index': 0, 'text': 'about us content for testing', 'embedding': [0.1] * 1536}]
_SPARSE = [{'indices': [1, 2], 'values': [0.5, 0.3]}]


class TestURLParsing:
    def test_valid_urls_queued_with_url_ids(self, client):
        _mock_db.create_url.return_value = 'url-abc'
        with patch('threading.Thread') as MockThread:
            MockThread.return_value = MagicMock()
            resp = client.post('/parse', json={'urls': ['https://example.com/about']})
        assert resp.status_code == 202
        data = resp.get_json()
        assert len(data['queued']) == 1
        assert data['queued'][0]['url'] == 'https://example.com/about'
        assert data['queued'][0]['url_id'] == 'url-abc'

    def test_empty_url_list_returns_400(self, client):
        resp = client.post('/parse', json={'urls': []})
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_webpage_status_endpoint(self, client):
        _mock_db.get_webpage_status.return_value = {'status': 'processing', 'in_kb': False}
        resp = client.get('/webpage/url-abc/status')
        assert resp.status_code == 200


class TestCrawlAndEmbedJob:
    def test_happy_path_marks_url_as_in_kb(self):
        _mock_db.get_all_chunks.return_value = []
        mock_encoder = MagicMock()
        with (
            patch('app.crawl_url', new=AsyncMock(return_value='## About Us\nWe do marketing.')),
            patch('app.embed_chunks', return_value=_EMBEDDED),
            patch('app.get_or_create_index', return_value=MagicMock()),
            patch('app.upsert_chunks'),
            patch('app.bm25_enc.get_encoder', return_value=None),
            patch('app.bm25_enc.fit_and_save', return_value=mock_encoder),
            patch('app.bm25_enc.encode_documents', return_value=_SPARSE),
            patch('app.chunk_text_recursive', return_value=['about us content for testing']),
            patch.dict(os.environ, {'CHUNKING_STRATEGY': 'recursive_token'}),
        ):
            flask_app._crawl_and_embed_job('url-abc', 'https://example.com/about')

        _mock_db.set_url_in_kb.assert_called_once_with('url-abc')
        _mock_db.set_url_kb_status.assert_called_with('url-abc', 'success')
        _mock_db.set_url_crawl_status.assert_called_with('url-abc', 'success')

    def test_chunks_inserted_and_upserted_to_pinecone(self):
        _mock_db.get_all_chunks.return_value = []
        mock_encoder = MagicMock()
        with (
            patch('app.crawl_url', new=AsyncMock(return_value='## About Us\nWe do marketing.')),
            patch('app.embed_chunks', return_value=_EMBEDDED),
            patch('app.get_or_create_index', return_value=MagicMock()),
            patch('app.upsert_chunks') as mock_upsert,
            patch('app.bm25_enc.get_encoder', return_value=None),
            patch('app.bm25_enc.fit_and_save', return_value=mock_encoder),
            patch('app.bm25_enc.encode_documents', return_value=_SPARSE),
            patch('app.chunk_text_recursive', return_value=['about us content for testing']),
            patch.dict(os.environ, {'CHUNKING_STRATEGY': 'recursive_token'}),
        ):
            flask_app._crawl_and_embed_job('url-abc', 'https://example.com')

        _mock_db.insert_webpage_chunks.assert_called_once()
        mock_upsert.assert_called_once()

    def test_crawl_failure_sets_failed_status_and_records_error(self):
        with (
            patch('app.crawl_url', new=AsyncMock(side_effect=Exception('Network error'))),
            patch.dict(os.environ, {'CHUNKING_STRATEGY': 'recursive_token'}),
        ):
            flask_app._crawl_and_embed_job('url-abc', 'https://example.com')

        _mock_db.set_url_crawl_status.assert_called_with('url-abc', 'failed')
        _mock_db.set_url_error.assert_called_once()
        args = _mock_db.set_url_error.call_args[0]
        assert 'Network error' in args[1]


# ---------------------------------------------------------------------------
# Phase 4 — RAG retrieval endpoints
# ---------------------------------------------------------------------------

class TestQueryKB:
    def _make_pinecone_mock(self, index_name='test-index'):
        mock_idx = MagicMock()
        mock_idx.name = index_name
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = [mock_idx]
        mock_pc.Index.return_value = MagicMock()
        return mock_pc

    def test_index_exists_returns_results(self, client):
        fake_results = [{'text': 'chunk', 'score': 0.9, 'doc_id': 'doc-1', 'chunk_id': 0}]
        with (
            patch('pinecone.Pinecone', return_value=self._make_pinecone_mock()),
            patch('app.hybrid_retrieve', return_value=fake_results),
        ):
            resp = client.post('/query', json={'query': 'what services do you offer'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'results' in data
        assert len(data['results']) == 1

    def test_no_pinecone_index_returns_empty_results(self, client):
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []
        with patch('pinecone.Pinecone', return_value=mock_pc):
            resp = client.post('/query', json={'query': 'test'})
        assert resp.status_code == 200
        assert resp.get_json()['results'] == []


class TestTestRAG:
    def _make_pinecone_mock(self, index_name='test-index'):
        mock_idx = MagicMock()
        mock_idx.name = index_name
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = [mock_idx]
        mock_pc.Index.return_value = MagicMock()
        return mock_pc

    def test_valid_query_returns_ranked_results(self, client):
        fake_results = [{'text': 'about us', 'score': 0.95, 'doc_id': 'doc-1', 'chunk_id': 0}]
        with (
            patch('pinecone.Pinecone', return_value=self._make_pinecone_mock()),
            patch('app.hybrid_retrieve', return_value=fake_results),
        ):
            resp = client.post('/test-rag', json={'query': 'what do you do'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'results' in data
        result = data['results'][0]
        assert 'rank' in result
        assert 'score' in result
        assert 'text' in result
        assert 'doc_id' in result

    def test_empty_query_returns_400(self, client):
        resp = client.post('/test-rag', json={'query': ''})
        assert resp.status_code == 400
        assert 'error' in resp.get_json()


# ---------------------------------------------------------------------------
# Phase 5 — End-to-end chatbot flows (sales controller + booking)
# ---------------------------------------------------------------------------

class TestChatbotRAGFlow:
    def test_rag_action_triggers_retrieve_relevant_chunks(self, client):
        _setup_chatbot_mocks(action='ANSWER_WITH_RAG', requires_rag=True)
        with (
            patch('app.retrieve_relevant_chunks', return_value='KB context here') as mock_rag,
            patch('app.generate_final_response', return_value='Based on our services...'),
        ):
            resp = client.post('/api', json={'message': 'what services?', 'session_id': 'sess-1'})
        assert resp.status_code == 200
        mock_rag.assert_called_once()

    def test_rag_context_passed_to_response_generator(self, client):
        _setup_chatbot_mocks(action='ANSWER_WITH_RAG', requires_rag=True)
        with (
            patch('app.retrieve_relevant_chunks', return_value='SEO and social media'),
            patch('app.generate_final_response', return_value='We do SEO.') as mock_gen,
        ):
            client.post('/api', json={'message': 'what do you offer', 'session_id': 'sess-1'})
        assert mock_gen.call_args.kwargs['rag_context'] == 'SEO and social media'


class TestChatbotBookingFlow:
    def test_booking_action_triggers_book_appointment(self, client):
        _setup_chatbot_mocks(action='OFFER_AVAILABLE_SLOTS', requires_booking=True)
        with (
            patch('app.book_appointment', return_value={'status': 'slots_available', 'slots': []}) as mock_book,
            patch('app.generate_final_response', return_value='Here are some slots...'),
        ):
            resp = client.post('/api', json={'message': 'book a call', 'session_id': 'sess-1'})
        assert resp.status_code == 200
        mock_book.assert_called_once()

    def test_booking_result_passed_to_response_generator(self, client):
        slots = {'status': 'slots_available', 'slots': [{'date': 'Monday', 'time': '12:00'}]}
        _setup_chatbot_mocks(action='OFFER_AVAILABLE_SLOTS', requires_booking=True)
        with (
            patch('app.book_appointment', return_value=slots),
            patch('app.generate_final_response', return_value='How about Monday?') as mock_gen,
        ):
            client.post('/api', json={'message': 'book a call', 'session_id': 'sess-1'})
        assert mock_gen.call_args.kwargs['booking_result'] == slots


class TestChatbotConversationHistory:
    def test_history_retrieved_using_correct_session_id(self, client):
        _setup_chatbot_mocks(history=[{'role': 'user', 'content': 'hi'}])
        with patch('app.generate_final_response', return_value='Hello again!'):
            client.post('/api', json={'message': 'follow up', 'session_id': 'sess-xyz'})
        _mock_db.get_conversation_history.assert_called_with('sess-xyz', limit=10)

    def test_guardrail_threshold_still_returns_200(self, client):
        _setup_chatbot_mocks(sales_turn=3)
        with patch('app.generate_final_response', return_value='Can I book a call for you?'):
            resp = client.post('/api', json={'message': 'tell me more', 'session_id': 'sess-1'})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Phase 6 — Text extraction lifecycle (async job run synchronously)
# ---------------------------------------------------------------------------

class TestExtractionJobLifecycle:
    def test_successful_extraction_transitions_to_success(self, client):
        _mock_db.get_status.return_value = DocumentStatus.CREATED
        _mock_db.get_file_path.return_value = ('test-bucket', 'doc-1.pdf')

        with (
            patch('threading.Thread', _SyncThread),
            patch('app.extract_doc_info', return_value='Extracted document text.'),
        ):
            resp = client.post('/extract-text', json={'doc_id': 'doc-1'})

        assert resp.status_code == 200
        success_calls = [
            c for c in _mock_db.transition_status.call_args_list
            if c.kwargs.get('new_status') == DocumentStatus.SUCCESS
        ]
        assert len(success_calls) == 1
        _mock_db.set_extraction_text_path.assert_called_once()

    def test_extraction_generic_exception_transitions_to_failed(self, client):
        _mock_db.get_status.return_value = DocumentStatus.CREATED
        _mock_db.get_file_path.return_value = ('test-bucket', 'doc-1.pdf')

        with (
            patch('threading.Thread', _SyncThread),
            patch('app.extract_doc_info', side_effect=Exception('OpenAI error')),
        ):
            resp = client.post('/extract-text', json={'doc_id': 'doc-1'})

        assert resp.status_code == 200
        failed_calls = [
            c for c in _mock_db.transition_status.call_args_list
            if c.kwargs.get('new_status') == DocumentStatus.FAILED
        ]
        assert len(failed_calls) >= 1

    def test_extraction_timeout_transitions_to_failed(self, client):
        _mock_db.get_status.return_value = DocumentStatus.CREATED
        _mock_db.get_file_path.return_value = ('test-bucket', 'doc-1.pdf')

        with (
            patch('threading.Thread', _SyncThread),
            patch('app.extract_doc_info', side_effect=ex.ExtractionTimeOut('Timed out')),
        ):
            resp = client.post('/extract-text', json={'doc_id': 'doc-1'})

        assert resp.status_code == 200
        failed_calls = [
            c for c in _mock_db.transition_status.call_args_list
            if c.kwargs.get('new_status') == DocumentStatus.FAILED
        ]
        assert len(failed_calls) >= 1
