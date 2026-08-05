"""Pytest fixtures for MemoryBot.

A fresh SQLite database is used so tests never touch the real data.
Environment variables are set BEFORE importing the application so that
config.Settings picks them up at import time.
"""
import os
import secrets
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

_TEST_DB_DIR = tempfile.mkdtemp(prefix="mb_test_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_memorybot.db")

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["SECRET_KEY"] = secrets.token_urlsafe(48)
os.environ["ENCRYPTION_KEY"] = secrets.token_urlsafe(48)
os.environ["OPENAI_API_KEY"] = ""
os.environ["ELEVENLABS_API_KEY"] = ""
os.environ["ADMIN_EMAIL"] = ""
os.environ["ADMIN_PASSWORD"] = ""
os.environ["UPLOAD_DIR"] = os.path.join(_TEST_DB_DIR, "uploads")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c


def _unique_email(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}@example.com"


def register(client, prefix: str, password: str = "test-password-123"):
    """Register a new user and return credentials. Logs the client in as them."""
    email = _unique_email(prefix)
    resp = client.post(
        "/auth/register",
        data={"name": "Test User", "email": email, "password": password},
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text
    assert "session_token" in resp.cookies
    return {"email": email, "password": password}


def login(client, email: str, password: str):
    resp = client.post(
        "/auth/login",
        data={"email": email, "password": password},
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text


def create_profile(client, name: str = "Test Person") -> str:
    resp = client.post(
        "/profiles/create",
        data={
            "name": name,
            "description": "A test memory profile.",
            "relationship": "Friend",
            "date_of_birth": "1950-01-01",
            "voice_id": "",
            "personality_traits": "",
            "favorite_phrases": "",
            "interests": "",
            "speaking_style": "",
            "writing_style": "",
            "values": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text
    location = resp.headers.get("location", "")
    assert location.startswith("/profiles/")
    return location.split("/profiles/")[1]


def create_admin_user():
    """Create a real administrator account directly in the test DB."""
    from database import SessionLocal, User
    from auth import hash_password
    email = f"admin-{secrets.token_hex(4)}@example.com"
    password = "admin-pass-123"
    db = SessionLocal()
    try:
        u = User(name="Admin", email=email, password_hash=hash_password(password), is_admin=True)
        db.add(u)
        db.commit()
        db.refresh(u)
    finally:
        db.close()
    return {"email": email, "password": password}


def upload_text(client, profile_id: str, name: str = "letter.txt", content: str = "Hello from the preserved letter."):
    """Upload a text file via the API and return its file_id."""
    resp = client.post(
        f"/profiles/{profile_id}/upload",
        files={"file": (name, content.encode("utf-8"), "text/plain")},
        data={"caption": "test", "memory_date": "1980-05-01"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("success"), data
    return data["file_id"]
