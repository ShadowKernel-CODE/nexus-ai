import json
import os
import asyncio
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from gradio_client import Client

from database import get_db, MemoryProfile, MemoryFile
from auth import get_user_from_request
from config import settings

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/tts")
async def text_to_speech(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    text = body.get("text", "")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        client = Client("neuphonic/neutts-2e", hf_token=settings.HF_TOKEN or None)
        result = client.predict(
            gen_text=text[:5000],
            speaker="emily",
            emotion="surprised",
            temperature=1,
            top_k=50,
            api_name="/infer",
        )
        audio_path = result

        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="speech.wav",
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/stt")
async def speech_to_text(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from fastapi import UploadFile, File
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
    voices = [
        {"id": "en-US-AvaNeural", "name": "Ava (Female, US)"},
        {"id": "en-US-AndrewNeural", "name": "Andrew (Male, US)"},
        {"id": "en-US-EmmaNeural", "name": "Emma (Female, US)"},
        {"id": "en-US-BrianNeural", "name": "Brian (Male, US)"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia (Female, UK)"},
        {"id": "en-GB-RyanNeural", "name": "Ryan (Male, UK)"},
        {"id": "en-AU-NatashaNeural", "name": "Natasha (Female, AU)"},
        {"id": "en-AU-WilliamNeural", "name": "William (Male, AU)"},
    ]
    return JSONResponse(voices)
