import os
import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from database import get_db, MemoryProfile
from auth import get_user_from_request
from config import settings

router = APIRouter(prefix="/voice", tags=["voice"])

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

PRESET_VOICES = [
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    {"id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    {"id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam"},
]


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
            {"error": "ElevenLabs API key not configured. Add ELEVENLABS_API_KEY to .env"},
            status_code=500,
        )

    voice_id = DEFAULT_VOICE_ID
    if profile_id:
        profile = db.query(MemoryProfile).filter(
            MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
        ).first()
        if profile and profile.voice_id:
            voice_id = profile.voice_id

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
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
        if resp.status_code != 200:
            return JSONResponse(
                {"error": f"ElevenLabs TTS error ({resp.status_code}): {resp.text[:300]}"},
                status_code=500,
            )
        return Response(content=resp.content, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/stt")
async def speech_to_text(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        return JSONResponse({"error": "STT not configured - no API key"}, status_code=500)

    form = await request.form()
    audio_file = form.get("audio")
    if not audio_file:
        return JSONResponse({"error": "No audio provided"}, status_code=400)

    temp_path = os.path.join(settings.UPLOAD_DIR, "stt_input.webm")
    content = await audio_file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
        with open(temp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return JSONResponse({"text": result.text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/voices")
async def list_voices():
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
