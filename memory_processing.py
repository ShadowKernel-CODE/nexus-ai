"""Multimodal memory ingestion pipeline.

Uploaded files are processed in a background thread so the UI can show live
processing states: uploading -> extracting/transcribing/analyzing -> indexing -> ready.
"""
import json
import os
import threading

from config import settings
from text_extractor import extract_text_from_file, chunk_text
from rag import (
    generate_embedding,
    transcribe_audio_file,
    describe_image_file,
    extract_audio_from_video,
)

from database import SessionLocal, MemoryFile, MemoryEmbedding

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".webm"}
VIDEO_EXTS = {".mp4", ".webm", ".mov"}
DOC_EXTS = {".pdf", ".docx", ".txt"}

ACTIVE_STATUSES = ("uploading", "extracting", "transcribing", "analyzing", "indexing")


def classify_memory_type(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "photograph"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in DOC_EXTS:
        return "document"
    return "document"


def user_safe_error(exc: Exception) -> str:
    """Return a user-facing error message that never exposes internals, paths, or secrets."""
    name = type(exc).__name__
    return f"Processing failed ({name}). Please check the file and try again."


def combine_image_text(description: str, caption: str, original_name: str) -> str:
    parts = []
    if description:
        parts.append(f"[Photograph: {original_name}]")
    if caption:
        parts.append(f"User context: {caption}")
    if description:
        parts.append(description)
    return "\n\n".join(parts)


def set_file_fields(db, file_id: str, **fields):
    file_obj = db.query(MemoryFile).filter(MemoryFile.id == file_id).first()
    if not file_obj:
        return
    for key, value in fields.items():
        setattr(file_obj, key, value)
    db.commit()


def process_memory_file(file_id: str) -> None:
    db = SessionLocal()
    try:
        file_obj = db.query(MemoryFile).filter(MemoryFile.id == file_id).first()
        if not file_obj:
            return

        ext = (file_obj.file_type or "").lower()
        memory_type = classify_memory_type(ext)
        file_obj.memory_type = memory_type
        file_obj.status = "extracting"
        db.commit()

        file_path = os.path.join(settings.UPLOAD_DIR, file_obj.filename)
        details = {}
        text = ""
        error_message = ""

        if memory_type == "document":
            text, chunks = extract_text_from_file(file_path, ext)
            file_obj.extracted_text = text
            file_obj.word_count = len(text.split())
        elif memory_type == "photograph":
            file_obj.status = "analyzing"
            db.commit()
            description = ""
            try:
                description = describe_image_file(
                    file_path, ext, file_obj.caption or "",
                    context=file_obj.profile.name if file_obj.profile else "",
                )
                details["vision"] = bool(description)
            except Exception as e:
                error_message = user_safe_error(e)
                details["vision"] = False
            file_obj.vision_description = description
            text = combine_image_text(description, file_obj.caption or "", file_obj.original_name or "")
            file_obj.extracted_text = text
            file_obj.word_count = len(text.split())
        elif memory_type == "audio":
            file_obj.status = "transcribing"
            db.commit()
            transcript = ""
            try:
                transcript = transcribe_audio_file(file_path)
                details["transcribed"] = bool(transcript)
            except Exception as e:
                error_message = user_safe_error(e)
                details["transcribed"] = False
            file_obj.transcript = transcript
            text = transcript or (file_obj.caption or "")
            file_obj.extracted_text = text
            file_obj.word_count = len(text.split())
        elif memory_type == "video":
            file_obj.status = "transcribing"
            db.commit()
            transcript = ""
            try:
                audio_tmp = os.path.join(settings.UPLOAD_DIR, f"{file_id}_audio_tmp.wav")
                extract_audio_from_video(file_path, audio_tmp)
                try:
                    transcript = transcribe_audio_file(audio_tmp)
                finally:
                    if os.path.exists(audio_tmp):
                        os.remove(audio_tmp)
                details["transcribed"] = bool(transcript)
                details["frames_analyzed"] = 0
            except Exception as e:
                error_message = user_safe_error(e)
                details["transcribed"] = False
            file_obj.transcript = transcript
            text = transcript or (file_obj.caption or "")
            file_obj.extracted_text = text
            file_obj.word_count = len(text.split())

        if not (text or "").strip():
            file_obj.status = "failed" if error_message else "ready"
            file_obj.error_message = error_message
            file_obj.is_processed = True
            file_obj.chunk_count = 0
            file_obj.processing_details = json.dumps(details)
            db.commit()
            return

        chunks = chunk_text(text)
        file_obj.text_chunks = json.dumps(chunks)
        file_obj.chunk_count = len(chunks)
        file_obj.error_message = error_message
        file_obj.status = "indexing"
        db.commit()

        db.query(MemoryEmbedding).filter(MemoryEmbedding.file_id == file_id).delete()
        db.commit()

        for i, chunk in enumerate(chunks):
            emb_vec = generate_embedding(chunk)
            emb = MemoryEmbedding(
                profile_id=file_obj.profile_id,
                file_id=file_id,
                content=chunk,
                embedding=json.dumps(emb_vec) if emb_vec else "[]",
                chunk_index=i,
            )
            db.add(emb)
        db.commit()

        file_obj.status = "ready"
        file_obj.is_processed = True
        file_obj.processing_details = json.dumps(details)
        db.commit()
    except Exception as e:
        try:
            f = db.query(MemoryFile).filter(MemoryFile.id == file_id).first()
            if f:
                f.status = "failed"
                f.error_message = user_safe_error(e)
                f.is_processed = True
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


def start_processing(file_id: str) -> None:
    thread = threading.Thread(target=process_memory_file, args=(file_id,), daemon=True)
    thread.start()


def recover_stale_processing() -> None:
    """On startup, mark files left mid-processing (e.g. from a crash) as failed."""
    db = SessionLocal()
    try:
        files = (
            db.query(MemoryFile)
            .filter(MemoryFile.status.in_(list(ACTIVE_STATUSES)))
            .all()
        )
        for f in files:
            f.status = "failed"
            f.error_message = "Processing was interrupted. Please re-upload the file."
            f.is_processed = True
        db.commit()
        if files:
            print(f"Recovered {len(files)} interrupted processing job(s).")
    finally:
        db.close()
