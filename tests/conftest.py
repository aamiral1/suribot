import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test that requires live services (PostgreSQL, OpenAI, Pinecone)",
    )
