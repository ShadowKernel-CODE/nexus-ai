"""Authorization tests: User B must never access User A's resources.

Covers profiles, files, conversations, messages, timeline, search,
downloads, deletion, and admin routes — including direct URL manipulation.
"""
import os
import re
import secrets

from tests.conftest import (
    register,
    login,
    create_profile,
    upload_text,
)


def _sse_conversation_id(resp) -> str:
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if '"done"' in payload and '"conversation_id"' in payload:
                match = re.search(r'"conversation_id":\s*"([^"]+)"', payload)
                if match:
                    return match.group(1)
    raise AssertionError(f"no conversation_id in SSE stream: {resp.text[:500]}")


def _wait_until_ready(client, profile_id, file_id, timeout=20):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/profiles/{profile_id}/files/{file_id}/status")
        if resp.status_code == 200:
            status = resp.json().get("status")
            if status == "ready":
                return True
            if status == "failed":
                return False
        time.sleep(0.4)
    return False


def test_cross_user_profile_access(client):
    a = register(client, "alice")
    profile_a = create_profile(client, "Alice Person")

    b = register(client, "bob")
    login(client, b["email"], b["password"])

    # Direct URL manipulation on another user's profile.
    resp = client.get(f"/profiles/{profile_a}", follow_redirects=False)
    assert resp.status_code != 200, "must not render another user's profile"
    assert "Alice Person" not in resp.text

    resp = client.get(f"/profiles/{profile_a}/timeline", follow_redirects=False)
    assert resp.status_code != 200

    resp = client.get(f"/profiles/api/{profile_a}")
    assert resp.status_code == 404

    resp = client.post(
        f"/profiles/{profile_a}/update",
        data={"name": "Hacked", "description": "", "relationship": ""},
        follow_redirects=False,
    )
    assert resp.status_code != 200, "must not update another user's profile"

    resp = client.post(f"/profiles/{profile_a}/delete", follow_redirects=False)
    assert resp.status_code != 200

    # Owner still has access.
    login(client, a["email"], a["password"])
    resp = client.get(f"/profiles/{profile_a}")
    assert resp.status_code == 200
    assert "Alice Person" in resp.text


def test_cross_user_file_access(client):
    a = register(client, "alice")
    profile_a = create_profile(client, "Alice Person")
    content = "private secret: the summer we went fishing by the lake"
    file_a = upload_text(client, profile_a, name="letter.txt", content=content)
    assert _wait_until_ready(client, profile_a, file_a)

    b = register(client, "bob")
    login(client, b["email"], b["password"])

    # Direct URL manipulation on another user's file.
    resp = client.get(f"/profiles/{profile_a}/files/{file_a}/status")
    assert resp.status_code == 404, "must not see another user's file status"

    resp = client.get(f"/profiles/{profile_a}/files/{file_a}/download")
    assert resp.status_code == 404, "must not download another user's file"
    assert "private secret" not in resp.text

    resp = client.get(f"/profiles/{profile_a}/files/{file_a}/view")
    assert resp.status_code == 404

    resp = client.post(f"/profiles/{profile_a}/files/{file_a}/delete", follow_redirects=False)
    assert resp.status_code != 200

    # Owner can still download their file.
    login(client, a["email"], a["password"])
    resp = client.get(f"/profiles/{profile_a}/files/{file_a}/download")
    assert resp.status_code == 200
    assert content.encode() in resp.content


def test_cross_user_conversation_access(client):
    a = register(client, "alice")
    profile_a = create_profile(client, "Alice Person")

    resp = client.post(
        "/chat/message",
        json={"profile_id": profile_a, "conversation_id": None, "content": "How did you meet grandpa?"},
    )
    conv_a = _sse_conversation_id(resp)

    b = register(client, "bob")
    login(client, b["email"], b["password"])

    resp = client.get(f"/chat/{profile_a}/conversation/{conv_a}", follow_redirects=False)
    assert resp.status_code != 200, "must not view another user's conversation"
    assert "How did you meet grandpa?" not in resp.text

    resp = client.post(
        "/chat/message",
        json={"profile_id": profile_a, "conversation_id": conv_a, "content": "trying to hijack"},
    )
    assert '"error"' in resp.text, "must not inject messages into another user's conversation"

    resp = client.post(f"/chat/{conv_a}/delete", follow_redirects=False)
    assert resp.status_code != 200

    # Owner still has their conversation.
    login(client, a["email"], a["password"])
    resp = client.get(f"/chat/{profile_a}/conversation/{conv_a}")
    assert resp.status_code == 200


def test_cross_user_search(client):
    a = register(client, "alice")
    profile_a = create_profile(client, "Alice Person")
    upload_text(client, profile_a, name="fishing.txt", content="we caught trout at the hidden lake bend")
    upload_text(client, profile_a, name="pie.txt", content="she baked apple pie every autumn")

    b = register(client, "bob")
    login(client, b["email"], b["password"])

    resp = client.get(f"/api/search", params={"q": "trout", "profile_id": profile_a})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("results", []) == [], "must not search another user's profile"

    resp = client.post("/search", data={"query": "trout", "profile_id": profile_a}, follow_redirects=False)
    assert resp.status_code == 200
    assert "hidden lake bend" not in resp.text, "must not leak another user's memory content"


def test_normal_user_cannot_access_admin(client):
    register(client, "mallory")
    resp = client.get("/admin", follow_redirects=False)
    assert resp.status_code == 302, "non-admin must be redirected away from admin"
    assert resp.headers.get("location", "").endswith("/auth/login")

    resp = client.get("/admin/api/stats")
    assert resp.status_code == 403, "non-admin must not read admin stats"

    resp = client.get("/admin/export")
    assert resp.status_code == 403, "non-admin must not export system data"

    resp = client.get("/admin/logs", follow_redirects=False)
    assert resp.status_code == 302, "non-admin must not view system logs"


def test_admin_can_access_admin(client):
    from tests.conftest import create_admin_user
    admin = create_admin_user()
    login(client, admin["email"], admin["password"])

    resp = client.get("/admin")
    assert resp.status_code == 200

    resp = client.get("/admin/api/stats")
    assert resp.status_code == 200
    assert "total_users" in resp.json()

    resp = client.get("/admin/export")
    assert resp.status_code == 200


def test_unauthenticated_access(client):
    client.cookies.clear()
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 302

    resp = client.get("/api/search")
    assert resp.status_code == 401

    resp = client.get("/profiles/whatever", follow_redirects=False)
    assert resp.status_code == 302

    client.cookies.clear()
