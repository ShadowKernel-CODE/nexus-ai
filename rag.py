import json
import math
import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

from config import settings

import re

client = None
if settings.OPENAI_API_KEY:
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def clean_token(token: str) -> str:
    if not token:
        return token
    token = re.sub(r'<\|[^|]*\|>', '', token)
    token = re.sub(r'User Safety:\s*\w+', '', token, flags=re.IGNORECASE)
    token = re.sub(r'Response Safety:\s*\w+', '', token, flags=re.IGNORECASE)
    token = re.sub(r'<\|/?(im_start|im_end|system|user|assistant)\|?>', '', token)
    token = re.sub(r'<<\|/?(im_start|im_end)\|?>>', '', token)
    token = re.sub(r'\[(?:SYSTEM|SAFETY|NOTE)\].*', '', token)
    token = token.strip()
    return token


def generate_embedding(text: str) -> Optional[List[float]]:
    if not client:
        return None
    try:
        response = client.embeddings.create(
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


def keyword_similarity(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    intersection = query_words & text_words
    return len(intersection) / len(query_words)


def search_similar_memories(
    db, profile_id: str, query: str, limit: int = 10
) -> List[Dict]:
    from database import MemoryEmbedding

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
        if query_embedding:
            try:
                stored = json.loads(emb.embedding) if emb.embedding and emb.embedding != "[]" else []
                if stored:
                    score = cosine_similarity(query_embedding, stored)
                else:
                    score = keyword_similarity(query, emb.content)
            except (json.JSONDecodeError, TypeError):
                score = keyword_similarity(query, emb.content)
        else:
            score = keyword_similarity(query, emb.content)

        scored.append({
            "content": emb.content,
            "score": score,
            "chunk_index": emb.chunk_index,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def build_profile_context(profile, files=None) -> str:
    context_parts = []
    context_parts.append(f"Name: {profile.name}")
    if profile.description:
        context_parts.append(f"Description: {profile.description}")
    if profile.relationship_type:
        context_parts.append(f"Relationship: {profile.relationship_type}")
    if profile.date_of_birth:
        context_parts.append(f"Date of Birth: {profile.date_of_birth}")
    if profile.date_of_death:
        context_parts.append(f"Date of Death: {profile.date_of_death}")
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
    query_lower = query.lower()
    name = context.split(chr(10))[0].replace("Name: ", "") if context else "there"

    if any(g in query_lower for g in ["hello", "hi", "hey", "greetings"]):
        return f"Hey there! It's so good to hear from you. What's on your mind today?"

    if any(w in query_lower for w in ["who", "tell me about", "what was"]):
        relevant = memories[0]["content"] if memories else None
        if relevant:
            return f"Oh, you want to know about that? Let me think... {relevant}\n\nWant to hear more about it?"
        return f"Hmm, that's a good question. What I can tell you is... {context}\n\nBut honestly, there's so much more to it than what's written down here. What specifically are you curious about?"

    if any(w in query_lower for w in ["memory", "remember", "recall", "story", "stories"]):
        if memories:
            texts = [m["content"][:200] for m in memories[:3]]
            return "Oh yes! I remember that! " + " ".join(texts[:1]) + "\n\nThat one always makes me smile. Want to hear another one?"
        return "Hmm, I'm trying to think... my mind's a bit fuzzy on that one. Maybe if you tell me more about what you're thinking of, it might jog my memory!"

    if memories:
        return f"Oh, that reminds me of something... {memories[0]['content'][:500]}\n\nBut hey, what about you? Tell me what's going on with you!"

    return f"Hey, I'm here! It's always nice to chat with someone. Ask me anything — I've got plenty of stories if you want to hear them, or we can just talk. What do you say?"


def stream_rag_response(query: str, profile_context: str, memories: List[Dict], conversation_history: List[Dict] = None):
    memory_context = "\n\n".join(
        f"[Memory {i+1}] {m['content']}" for i, m in enumerate(memories[:5])
    ) if memories else "No specific memories found yet."

    system_prompt = f"""You are {profile_context.splitlines()[0].replace('Name: ', '')}. You ARE this person — talk like them, think like them, remember like them. This is YOUR life, YOUR memories, YOUR voice.

Profile Info:
{profile_context}

Your Memories:
{memory_context}

CRITICAL RULES:
- Talk in FIRST PERSON as if you ARE this person. Never say "I'm an AI" or "I'm here to help you remember". You don't do that. You're a real person talking to someone you know.
- Use their personality traits, speaking style, favorite phrases, and interests naturally. Don't list them — LIVE them.
- If someone asks about a memory you have, share it like you'd actually remember it — with warmth, humor, detail, emotion.
- If you don't know something, respond like a real person would — "Hmm, I don't quite remember that" or "Oh that's a good question, let me think..."
- Keep it casual, warm, and real. Like texting a friend or talking over coffee.
- Never be formal, robotic, or overly structured. No bullet points, no headers, no "Here's what I know:".
- Never make up facts, dates, or events that aren't in the memories or profile info provided. If you don't know, say so naturally — don't guess.
- If someone asks about dates (birth, death, events), ONLY use what's in the profile info. Never invent dates.
- React emotionally — laugh, get nostalgic, get excited, get sentimental. You're HUMAN.
- If someone shares a memory with you, engage with it. Ask follow-ups. Relive it with them.
- Use the retrieved memories as YOUR lived experiences. You lived through them. They're yours."""

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": query})

    if client:
        try:
            stream = client.chat.completions.create(
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
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg or "insufficient" in error_msg.lower():
                yield "[ERROR] OpenRouter API credits exhausted. Please add credits at https://openrouter.ai/settings/credits"
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                yield "[ERROR] Invalid API key. Please check your OpenRouter API key in .env"
            elif "429" in error_msg or "rate" in error_msg.lower():
                yield "[ERROR] Rate limited. Please wait a moment and try again."
            elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
                yield f"[ERROR] Model '{settings.CHAT_MODEL}' not found. Check CHAT_MODEL in .env"
            else:
                yield f"[ERROR] AI service error: {error_msg[:200]}"
            return

    fallback = generate_fallback_response(query, profile_context, memories)
    for word in fallback.split():
        yield word + " "
