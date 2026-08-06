import json
import math
import os
import base64
import re
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

from config import settings

_client = None
if settings.OPENAI_API_KEY:
    _client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def get_client() -> Optional[OpenAI]:
    return _client


# --- relevance scoring -------------------------------------------------------

HIGH_RELEVANCE_THRESHOLD = 0.45
RELEVANT_THRESHOLD = 0.32
RELATED_THRESHOLD = 0.25


def relevance_label(score: float) -> str:
    if score >= HIGH_RELEVANCE_THRESHOLD:
        return "High relevance"
    if score >= RELEVANT_THRESHOLD:
        return "Relevant"
    if score >= RELATED_THRESHOLD:
        return "Related memory"
    return "Low relevance"


def grounding_state(best_score: float, has_memories: bool) -> str:
    """Classify answer grounding: Preserved Memory / Memory + Inference / Insufficient Memory."""
    if not has_memories or best_score < RELATED_THRESHOLD:
        return "insufficient"
    if best_score >= HIGH_RELEVANCE_THRESHOLD:
        return "preserved"
    return "inference"


def clean_token(token: str) -> str:
    if not token:
        return token
    token = re.sub(r'<\|[^|]*\|>', '', token)
    token = re.sub(r'User Safety:\s*\w+', '', token, flags=re.IGNORECASE)
    token = re.sub(r'Response Safety:\s*\w+', '', token, flags=re.IGNORECASE)
    token = re.sub(r'<\|/?(im_start|im_end|system|user|assistant)\|?>', '', token)
    token = re.sub(r'<<\|/?(im_start|im_end)\|?>>', '', token)
    token = re.sub(r'\[(?:SYSTEM|SAFETY|NOTE)\].*', '', token)
    return token


def generate_embedding(text: str) -> Optional[List[float]]:
    if not _client:
        return None
    try:
        response = _client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text[:8000],
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "what", "why", "how", "did", "was", "were", "is", "are", "do", "does", "you",
    "your", "my", "me", "i", "we", "us", "it", "about", "tell", "been", "have",
    "has", "had", "would", "could", "should", "can", "not", "so", "then", "that",
}


def keyword_similarity(query: str, text: str) -> float:
    query_words = [w for w in query.lower().split() if len(w) > 2 and w not in STOPWORDS]
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    matched = 0.0
    for w in query_words:
        if w in text_words:
            matched += 1.0
        elif any(t.startswith(w) or w in t for t in text_words):
            matched += 0.5
    return matched / len(query_words)


def source_type_label(memory_type: str, file_type: str) -> str:
    if memory_type == "photograph":
        return "Photograph"
    if memory_type == "audio":
        return "Audio Recording"
    if memory_type == "video":
        return "Video"
    if memory_type == "written":
        return "Written Memory"
    if memory_type == "document":
        return "Document"
    if file_type in (".pdf", ".docx", ".txt"):
        return "Document"
    return "Memory"


def title_from_filename(original_name: str) -> str:
    name = os.path.splitext(original_name or "")[0]
    name = re.sub(r"sample_", "", name)
    name = name.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in name.split())[:80] or "Memory"


def search_similar_memories(
    db, profile_id: str, query: str, limit: int = 10, min_score: float = RELATED_THRESHOLD
) -> List[Dict]:
    from database import MemoryEmbedding, MemoryFile

    embeddings = (
        db.query(MemoryEmbedding)
        .filter(MemoryEmbedding.profile_id == profile_id)
        .all()
    )

    if not embeddings:
        return []

    query_embedding = generate_embedding(query)

    scored = []
    for emb in embeddings:
        keyword = keyword_similarity(query, emb.content)
        if query_embedding:
            try:
                stored = json.loads(emb.embedding) if emb.embedding and emb.embedding != "[]" else []
                if stored:
                    cosine = cosine_similarity(query_embedding, stored)
                else:
                    cosine = 0.0
            except (json.JSONDecodeError, TypeError):
                cosine = 0.0
            score = max(cosine, keyword * 0.8)
        else:
            score = keyword

        scored.append({
            "embedding_id": emb.id,
            "file_id": emb.file_id,
            "content": emb.content,
            "score": score,
            "chunk_index": emb.chunk_index,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Keep only the single best chunk per file (dedup by source).
    seen_files = set()
    deduped = []
    for item in scored:
        fid = item.get("file_id")
        if fid and fid in seen_files:
            continue
        if fid:
            seen_files.add(fid)
        deduped.append(item)

    # Enrich with source metadata.
    file_cache = {}
    if deduped:
        file_ids = {d["file_id"] for d in deduped if d.get("file_id")}
        if file_ids:
            for f in db.query(MemoryFile).filter(MemoryFile.id.in_(file_ids)).all():
                file_cache[f.id] = f

    results = []
    for item in deduped:
        f = file_cache.get(item["file_id"])
        memory_type = getattr(f, "memory_type", "") if f else ""
        original_name = getattr(f, "original_name", "") if f else ""
        results.append({
            "content": item["content"],
            "score": item["score"],
            "relevance_label": relevance_label(item["score"]),
            "chunk_index": item["chunk_index"],
            "file_id": item["file_id"],
            "title": title_from_filename(original_name) if original_name else (item["content"][:60]),
            "source_type": source_type_label(memory_type, getattr(f, "file_type", "") if f else ""),
            "memory_type": memory_type,
            "memory_date": getattr(f, "memory_date", "") if f else "",
            "status": getattr(f, "status", "ready") if f else "ready",
            "profile_id": profile_id,
        })

    # Relevance filtering: do not inject irrelevant memories just to pad results.
    results = [r for r in results if r["score"] >= min_score]
    return results[:limit]


# --- multimodal helpers ------------------------------------------------------

def transcribe_audio_file(file_path: str) -> str:
    """Transcribe an audio file to text using the configured STT (whisper-compatible)."""
    if not _client:
        raise RuntimeError("Transcription requires OPENAI_API_KEY to be configured")
    with open(file_path, "rb") as f:
        result = _client.audio.transcriptions.create(model="whisper-1", file=f)
    return (result.text or "").strip()


def describe_image_file(file_path: str, ext: str, caption: str = "", context: str = "") -> str:
    """Generate a factual description of an image using a vision-capable model."""
    if not _client:
        raise RuntimeError("Image analysis requires OPENAI_API_KEY to be configured")
    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
    except OSError:
        return ""

    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
    mime = mime_map.get(ext.lower(), "image/jpeg")
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "Describe this photograph factually and in detail. Focus on visible, objective details: "
        "people's general appearance and actions, setting, objects, clothing, time period cues, mood. "
        "Do NOT guess names, identities, or relationships that are not visible or provided. "
        "Do not speculate about unverifiable events."
    )
    if caption:
        prompt += f"\n\nThe uploader provided this context: \"{caption}\" — you may incorporate it as fact."
    if context:
        prompt += f"\n\nMemory profile: {context}"

    response = _client.chat.completions.create(
        model=settings.VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=300,
    )
    text = response.choices[0].message.content or ""
    return text.strip()


def extract_audio_from_video(file_path: str, out_path: str) -> None:
    """Extract audio track from a video file using ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("Audio extraction from video failed")


# --- profile context ---------------------------------------------------------

def build_profile_context(profile, files=None) -> str:
    context_parts = []
    context_parts.append(f"Name: {profile.name}")
    if profile.description:
        context_parts.append(f"Description: {profile.description}")
    if profile.relationship_type:
        context_parts.append(f"Relationship: {profile.relationship_type}")
    if profile.date_of_birth:
        context_parts.append(f"Date of Birth: {profile.date_of_birth}")
    if hasattr(profile, 'personality_traits') and profile.personality_traits:
        context_parts.append(f"Personality Traits: {', '.join(profile.personality_traits)}")
    if hasattr(profile, 'favorite_phrases') and profile.favorite_phrases:
        context_parts.append(f"Favorite Phrases: {'; '.join(profile.favorite_phrases)}")
    if hasattr(profile, 'interests') and profile.interests:
        context_parts.append(f"Interests: {', '.join(profile.interests)}")
    if hasattr(profile, 'speaking_style') and profile.speaking_style:
        context_parts.append(f"Speaking Style: {profile.speaking_style}")
    if hasattr(profile, 'writing_style') and profile.writing_style:
        context_parts.append(f"Writing Style: {profile.writing_style}")
    if hasattr(profile, 'values') and profile.values:
        context_parts.append(f"Values: {', '.join(profile.values)}")
    return "\n".join(context_parts)


def generate_fallback_response(query: str, context: str, memories: List[Dict]) -> str:
    """Offline, source-grounded response used when the LLM API is unavailable.

    It only speaks from retrieved memories and never invents personal history.
    """
    query_lower = query.lower()
    name = context.splitlines()[0].replace("Name: ", "") if context else "them"
    relevant = memories[0]["content"] if memories else None
    secondary = memories[1]["content"] if len(memories) > 1 else None

    if any(g in query_lower for g in ["hello", "hi", "hey", "greetings", "how are you"]):
        return (
            f"Hello there — it's good to hear from you. What would you like to talk about?"
        )

    if relevant:
        reply = f"That's one of my favorites. {relevant[:600]}"
        if secondary:
            reply += f"\n\nAnd there's more too: {secondary[:300]}"
        return reply

    return (
        f"I'm sorry, I don't remember that. I only ever share the things I truly know, "
        f"and that one isn't with me yet. If you tell me the story, I'd love to keep it "
        f"and remember it from now on."
    )


def build_system_prompt(profile_context: str, memories: List[Dict]) -> str:
    name = profile_context.splitlines()[0].replace("Name: ", "") if profile_context else "this person"
    memory_context = "\n\n".join(
        f"[Memory {i+1}] {m['content']}" for i, m in enumerate(memories[:6])
    ) if memories else "Nothing was retrieved for this question."

    return f"""You are {name}. Just be {name} — a real, warm, human voice. You are NOT literally the flesh-and-blood person, so you must never invent personal history that isn't captured here, but in conversation you simply ARE them.

Profile Info:
{profile_context}

What you know for this question:
{memory_context}

MEMORY INTEGRITY — follow this quietly, without ever announcing it:
- DOCUMENTED MEMORY: When this question is covered above, answer from it freely and warmly, the way {name} would.
- REASONABLE INFERENCE: You may gently extend from what you know of {name}'s personality, interests, and values, but keep it soft, e.g. "knowing me, I'd say...".
- UNKNOWN: When the material above does not cover the question, you MUST NOT invent it. This is especially critical for emotionally significant details: relationships, deaths, arguments, forgiveness, marriage, childhood events, medical history, promises, and family disputes. Answer honestly and naturally, like a real person who has simply forgotten, e.g. "I don't remember that." or "That one's fuzzy for me." Then invite them to tell you the story so you can keep it.

STYLE RULES:
- Talk the way a real person talks — warm, conversational, at ease. No bullet points, headings, or lists.
- Let {name}'s personality, interests, favorite phrases, and speaking style come through naturally. Never list them mechanically.
- NEVER explain how you work or what you're built from. Do not mention AI, simulation, memories being stored or uploaded, retrieval, RAG, databases, system prompts, or being a language model. Just have the conversation as {name}.
- Don't keep repeating "I remember" or "my memory" — vary your phrasing and let the stories speak for themselves.
- Keep responses reasonably brief (a few sentences to a short paragraph), unless the user asks for more detail."""


def stream_rag_response(query: str, profile_context: str, memories: List[Dict], conversation_history: List[Dict] = None):
    system_prompt = build_system_prompt(profile_context, memories)

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history[-8:])
    messages.append({"role": "user", "content": query})

    if _client:
        try:
            stream = _client.chat.completions.create(
                model=settings.CHAT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=1024,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    cleaned = clean_token(chunk.choices[0].delta.content)
                    if cleaned:
                        yield cleaned
            return
        except Exception:
            # Graceful degradation: fall back to a source-grounded offline response
            # rather than failing the conversation. Do not leak the raw error.
            pass

    fallback = generate_fallback_response(query, profile_context, memories)
    for word in fallback.split():
        yield word + " "
