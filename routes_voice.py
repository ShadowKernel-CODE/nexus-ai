import os
import tempfile
import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from database import get_db, MemoryProfile
from auth import get_user_from_request
from config import settings
from rag import transcribe_audio_file

router = APIRouter(prefix="/voice", tags=["voice"])

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah - account voice that works on free plan

PRESET_VOICES = [
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah"},
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger"},
    {"id": "FGY2WhTYpPnrIDTdsKH5", "name": "Laura"},
    {"id": "IKne3meq5aSn9XLyUdCD", "name": "Charlie"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George"},
    {"id": "SAz9YHcvj6GT2YYXdXww", "name": "River"},
    {"id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
]


def _safe_error(exc: Exception) -> str:
    return f"Voice service error ({type(exc).__name__}). Please try again."


@router.post("/tts")
async def text_to_speech(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    text = body.get("text", "")
    profile_id = body.get("profile_id")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if not settings.ELEVENLABS_API_KEY:
        return JSONResponse(
            {"error": "Text-to-speech is not configured. Please add ELEVENLABS_API_KEY."},
            status_code=500,
        )

    voice_id = DEFAULT_VOICE_ID
    if profile_id:
        profile = db.query(MemoryProfile).filter(
            MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
        ).first()
        if profile and profile.voice_id:
            voice_id = profile.voice_id

    async def synthesize(vid: str):
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{ELEVENLABS_BASE_URL}/text-to-speech/{vid}",
                headers={
                    "xi-api-key": settings.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text[:5000],
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                },
            )
        return resp

    try:
        resp = await synthesize(voice_id)
        if resp.status_code != 200 and voice_id != DEFAULT_VOICE_ID:
            resp = await synthesize(DEFAULT_VOICE_ID)
        if resp.status_code != 200:
            return JSONResponse(
                {"error": "Text-to-speech failed. Please try again or choose another voice."},
                status_code=500,
            )
        return Response(content=resp.content, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": _safe_error(e)}, status_code=500)


@router.post("/stt")
async def speech_to_text(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not (settings.OPENAI_API_KEY or settings.ELEVENLABS_API_KEY):
        return JSONResponse(
            {"error": "Speech-to-text is not configured. Add ELEVENLABS_API_KEY or OPENAI_API_KEY."},
            status_code=500,
        )

    form = await request.form()
    audio_file = form.get("audio")
    if not audio_file:
        return JSONResponse({"error": "No audio provided"}, status_code=400)

    content = await audio_file.read()
    if len(content) > 25 * 1024 * 1024:
        return JSONResponse({"error": "Audio too large (max 25MB)"}, status_code=400)

    tmp_dir = tempfile.mkdtemp(prefix="mb_stt_")
    temp_path = os.path.join(tmp_dir, "input.webm")
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        text = transcribe_audio_file(temp_path)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": _safe_error(e)}, status_code=500)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


@router.get("/voices")
async def list_voices(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not settings.ELEVENLABS_API_KEY:
        return JSONResponse(PRESET_VOICES)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{ELEVENLABS_BASE_URL}/voices",
                headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
            )
        if resp.status_code == 200:
            data = resp.json()
            voices = [
                {"id": v["voice_id"], "name": v["name"]}
                for v in data.get("voices", [])
            ]
            if voices:
                return JSONResponse(voices)
    except Exception:
        pass
    return JSONResponse(PRESET_VOICES)
