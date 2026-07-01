"""
Shared pytest fixtures.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup on enter, shutdown on exit
        yield c
