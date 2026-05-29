"""Tests for admin document upload, URL ingestion, and KB status tracking."""
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
import botocore.exceptions
from enums import DocumentStatus

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

_404_error = botocore.exceptions.ClientError(
    {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'HeadObject'
)


@pytest.fixture(autouse=True)
def reset_mocks():
    flask_app.db = _mock_db
    flask_app.s3 = _mock_s3
    _mock_db.reset_mock()
    _mock_s3.reset_mock()
    # By default, S3 reports files as not found so uploads are not rejected
    _mock_s3.meta.client.head_object.side_effect = _404_error


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
# Admin dashboard access (FR10)
# ---------------------------------------------------------------------------

class TestAdminDashboardAccess:
    def test_admin_page_redirects_without_login(self, client):
        """Unauthenticated access to the admin dashboard is redirected (FR10)."""
        resp = client.get('/admin', follow_redirects=False)
        assert resp.status_code == 302

    def test_admin_page_loads_when_logged_in(self, logged_in_client):
        """Admin dashboard loads for a logged-in user (FR10)."""
        resp = logged_in_client.get('/admin')
        assert resp.status_code == 200

    def test_documents_endpoint_returns_list(self, logged_in_client):
        """/documents returns a JSON list of uploaded documents (FR10)."""
        _mock_db.get_documents.return_value = []
        resp = logged_in_client.get('/documents')
        assert resp.status_code == 200
        assert isinstance(resp.json.get('documents'), list)


# ---------------------------------------------------------------------------
# Document upload (FR11) and status tracking (FR13)
# ---------------------------------------------------------------------------

class TestDocumentUpload:
    def test_document_upload_creates_db_record(self, logged_in_client):
        """Uploading a document creates a database record (FR11)."""
        from io import BytesIO
        _mock_db.create.return_value = 'doc-001'

        data = {
            'file': (BytesIO(b'%PDF test content'), 'suri_services.pdf'),
            'doc_type': 'knowledge_base',
            'doc_structure': 'free_flow',
        }
        resp = logged_in_client.post('/admin', data=data, content_type='multipart/form-data')
        assert resp.status_code in (200, 302)
        _mock_db.create.assert_called_once()

    def test_uploaded_document_appears_in_document_list(self, logged_in_client):
        """An uploaded document shows up in the document list (FR11)."""
        # Route iterates rows as tuples: [0]=doc_id [1]=source_type [2]=file_name
        # [3]=size [4]=type [5]=date [6..7]=padding [8]=status [9..11]=padding [12]=in_kb
        row = ('doc-001', 'upload', 'suri_services.pdf', '10KB', 'PDF',
               '2026-05-28', None, None, 'created', None, None, None, False)
        _mock_db.get_all_documents.return_value = [row]
        resp = logged_in_client.get('/documents')
        assert resp.status_code == 200
        docs = resp.json['documents']
        assert len(docs) == 1
        assert docs[0]['file_name'] == 'suri_services.pdf'

    def test_document_status_tracked_through_processing(self, logged_in_client):
        """Document status can be polled via the status endpoint (FR13)."""
        _mock_db.get_kb_status.return_value = DocumentStatus.PROCESSING.value
        resp = logged_in_client.get('/document/doc-001/kb-status')
        assert resp.status_code == 200
        assert resp.json['kb_status'] == DocumentStatus.PROCESSING.value

    def test_completed_document_shows_success_status(self, logged_in_client):
        """A fully ingested document reports 'success' kb_status (FR13)."""
        _mock_db.get_kb_status.return_value = DocumentStatus.SUCCESS.value
        resp = logged_in_client.get('/document/doc-002/kb-status')
        assert resp.status_code == 200
        assert resp.json['kb_status'] == DocumentStatus.SUCCESS.value


# ---------------------------------------------------------------------------
# URL ingestion (FR12, FR13)
# ---------------------------------------------------------------------------

class TestURLIngestion:
    def test_valid_url_is_queued_for_crawling(self, logged_in_client):
        """A valid URL is queued for crawling and returns a URL ID (FR12)."""
        _mock_db.create_url.return_value = 'url-001'
        with patch('threading.Thread'):
            resp = logged_in_client.post('/parse', json={
                'urls': ['https://surimarketing.co.uk']
            })
        assert resp.status_code == 202
        data = resp.json
        assert 'queued' in data
        assert len(data['queued']) == 1

    def test_empty_url_list_returns_400(self, logged_in_client):
        """An empty URL list is rejected with HTTP 400 (FR12)."""
        resp = logged_in_client.post('/parse', json={'urls': []})
        assert resp.status_code == 400

    def test_url_processing_status_is_trackable(self, logged_in_client):
        """URL processing status is trackable via the status endpoint (FR13)."""
        _mock_db.get_webpage_status.return_value = {
            'crawl_status': 'processing',
            'kb_status': None,
            'error_msg': None,
        }
        resp = logged_in_client.get('/webpage/url-001/status')
        assert resp.status_code == 200
        assert resp.json['crawl_status'] == 'processing'
