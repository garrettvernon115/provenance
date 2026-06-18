"""API smoke tests via FastAPI TestClient.

The app's lifespan opens a real DB pool, so these skip when Postgres is down
(e.g. minimal CI). The full query→rerank→answer path is validated separately;
here we check the app wires up and the no-LLM endpoints behave.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app as app_mod


@pytest.fixture(scope="module")
def client():
    try:
        with TestClient(app_mod.app) as c:
            yield c
    except Exception as exc:  # noqa: BLE001 - pool open fails when DB is down
        pytest.skip(f"app/DB unavailable: {exc}")


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text


def test_empty_query_rejected(client):
    r = client.post("/api/query", json={"query": "   "})
    assert r.status_code == 400


def test_missing_document_404(client):
    r = client.get("/api/documents/does-not-exist")
    assert r.status_code == 404
