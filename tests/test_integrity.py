"""Memory Integrity & RAG tests.

SUPPORTED   -> grounded response with sources.
UNSUPPORTED -> the AI must NOT invent personal history.
Also covers retrieval thresholds, dedup, chunking, and system-prompt rules.
"""
import json
import re
import time

from tests.conftest import register, create_profile, upload_text


def _wait_until_ready(client, profile_id, file_id, timeout=20):
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


def _sse_parts(text):
    """Return a list of parsed SSE data payloads."""
    parts = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                parts.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return parts


def _last_done(payloads):
    for p in reversed(payloads):
        if p.get("done"):
            return p
    return None


def _meta(payloads):
    for p in payloads:
        if p.get("meta") is not None:
            return p["meta"]
    return None


def test_relevance_and_grounding_labels():
    from rag import relevance_label, grounding_state
    assert relevance_label(0.5) == "High relevance"
    assert relevance_label(0.35) == "Relevant"
    assert relevance_label(0.27) == "Related memory"
    assert grounding_state(0.5, True) == "preserved"
    assert grounding_state(0.35, True) == "inference"
    assert grounding_state(0.1, True) == "insufficient"
    assert grounding_state(0.0, False) == "insufficient"


def test_fallback_never_invents():
    from rag import generate_fallback_response

    context = "Name: Margaret Johnson"
    no_memories = []
    reply = generate_fallback_response(
        "What did you give me for my 10th birthday?", context, no_memories
    )
    assert "don't have a preserved memory" in reply.lower()
    for forbidden in ("bicycle", "red bicycle", "a red bike"):
        assert forbidden.lower() not in reply.lower()

    memories = [{"content": "Margaret met Robert at a church social in 1946."}]
    reply = generate_fallback_response("Where did you meet grandpa?", context, memories)
    assert "church social" in reply


def test_system_prompt_integrity_rules():
    from rag import build_system_prompt
    prompt = build_system_prompt("Name: Margaret Johnson", [])
    assert "MEMORY INTEGRITY" in prompt
    assert "NOT literally" in prompt
    assert "DOCUMENTED MEMORY" in prompt
    assert "UNKNOWN" in prompt
    assert "must not invent" in prompt.lower()


def test_chunking_bounds():
    from text_extractor import chunk_text
    text = (" ".join(["word"] * 5000))  # ~5000 words, definitely > chunk
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1, "long text must be split"
    assert all(len(c.split()) <= 1000 for c in chunks)


def test_retrieval_dedup_and_threshold(client):
    from database import SessionLocal, MemoryProfile
    from rag import search_similar_memories

    user = register(client, "retrieval")
    profile_id = create_profile(client, "Retrieval Person")

    ids = [
        upload_text(
            client, profile_id,
            name="meeting.txt",
            content="Margaret met Robert at a church social in 1946. They married in June 1948.",
        ),
        upload_text(
            client, profile_id,
            name="pie.txt",
            content="Margaret's apple pie used a secret blend of cinnamon and nutmeg.",
        ),
        upload_text(
            client, profile_id,
            name="garden.txt",
            content="The roses in the garden bloomed every June.",
        ),
    ]
    for fid in ids:
        assert _wait_until_ready(client, profile_id, fid), f"file {fid} did not finish processing"

    db = SessionLocal()
    try:
        profile = db.query(MemoryProfile).filter(MemoryProfile.id == profile_id).first()
        assert profile is not None

        results = search_similar_memories(db, profile_id, "church social robert")
        assert results, "expected a relevant memory for the church question"

        # Each source appears at most once (dedup by file).
        file_ids = [r["file_id"] for r in results]
        assert len(set(file_ids)) == len(file_ids)

        # Unsupported topic must be filtered out, not padded in.
        results = search_similar_memories(db, profile_id, "what did you give me for my 10th birthday")
        assert results == [], "irrelevant memories must not be injected"
    finally:
        db.close()


def test_supported_question_is_grounded(client):
    user = register(client, "grounded")
    profile_id = create_profile(client, "Grounded Person")
    ids = [
        upload_text(
            client, profile_id,
            name="meeting.txt",
            content="Margaret met Robert at a church social in 1946 and they married in June 1948.",
        ),
        upload_text(
            client, profile_id,
            name="pie.txt",
            content="Her apple pie recipe was a family secret.",
        ),
    ]
    for fid in ids:
        assert _wait_until_ready(client, profile_id, fid)

    resp = client.post(
        "/chat/message",
        json={"profile_id": profile_id, "conversation_id": None, "content": "Where did you and Robert meet?"},
    )
    assert resp.status_code == 200
    payloads = _sse_parts(resp.text)

    meta = _meta(payloads)
    assert meta is not None
    assert meta.get("sources"), "supported question must attach source metadata"
    assert meta.get("grounding") in ("preserved", "inference")

    full = " ".join(p.get("token", "") for p in payloads if p.get("token"))
    assert "church" in full.lower() or "1946" in full


def test_unsupported_question_not_fabricated(client):
    user = register(client, "unsupported")
    profile_id = create_profile(client, "Unsupported Person")
    fid = upload_text(
        client, profile_id,
        name="pie.txt",
        content="Margaret loved baking apple pie and grew roses in her garden.",
    )
    assert _wait_until_ready(client, profile_id, fid)

    resp = client.post(
        "/chat/message",
        json={"profile_id": profile_id, "conversation_id": None, "content": "What did you give me for my 10th birthday?"},
    )
    assert resp.status_code == 200
    payloads = _sse_parts(resp.text)

    meta = _meta(payloads)
    assert meta is not None
    assert meta.get("grounding") == "insufficient"

    full = " ".join(p.get("token", "") for p in payloads if p.get("token"))
    normalized = " ".join(full.split())
    for forbidden in ("red bicycle", "bicycle", "I gave you a"):
        assert forbidden.lower() not in normalized.lower(), "AI must not fabricate undocumented memories"
    assert "don't have a preserved memory" in normalized.lower() or "no preserved memory" in normalized.lower()
