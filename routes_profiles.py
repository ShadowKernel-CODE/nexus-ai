from fastapi import APIRouter, Request, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, Response
from sqlalchemy.orm import Session
from collections import defaultdict
from datetime import datetime
import os
import uuid
import json

from database import get_db, User, MemoryProfile, MemoryFile, MemoryEmbedding, Conversation, Message, AuditLog
from auth import get_user_from_request, get_profile_for_user, get_file_for_profile
from config import settings
from memory_processing import start_processing
from completeness import compute_completeness

router = APIRouter(prefix="/profiles", tags=["profiles"])

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def audit(db: Session, user_id, action, resource_type="", resource_id="", details=""):
    db.add(AuditLog(user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, details=details))
    db.commit()


def _get_owned_file(db, user, profile_id, file_id) -> MemoryFile:
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    file_obj = get_file_for_profile(db, file_id, profile_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")
    return file_obj


def _delete_file_from_disk(file_obj: MemoryFile):
    file_path = os.path.join(settings.UPLOAD_DIR, file_obj.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


@router.get("", response_class=HTMLResponse)
async def profiles_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).order_by(MemoryProfile.created_at.desc()).all()
    return templates.TemplateResponse(request, "profiles.html", {"request": request, "user": user, "profiles": profiles})


@router.post("/create")
async def create_profile(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    relationship: str = Form(""),
    date_of_birth: str = Form(""),
    voice_id: str = Form(""),
    personality_traits: str = Form(""),
    favorite_phrases: str = Form(""),
    interests: str = Form(""),
    speaking_style: str = Form(""),
    writing_style: str = Form(""),
    values: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    name = (name or "").strip()
    if not name:
        return RedirectResponse(url="/profiles", status_code=302)

    profile = MemoryProfile(
        user_id=user.id,
        name=name,
        description=description,
        relationship_type=relationship,
        date_of_birth=date_of_birth,
        voice_id=voice_id,
        personality_traits=[t.strip() for t in personality_traits.split(",") if t.strip()] if personality_traits else [],
        favorite_phrases=[p.strip() for p in favorite_phrases.split("\n") if p.strip()] if favorite_phrases else [],
        interests=[i.strip() for i in interests.split(",") if i.strip()] if interests else [],
        speaking_style=speaking_style,
        writing_style=writing_style,
        values=[v.strip() for v in values.split(",") if v.strip()] if values else [],
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    audit(db, user.id, "create_profile", resource_type="profile", resource_id=profile.id, details=f"Created profile: {name}")
    return RedirectResponse(url=f"/profiles/{profile.id}", status_code=302)


@router.get("/{profile_id}", response_class=HTMLResponse)
async def profile_detail(request: Request, profile_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)
    files = db.query(MemoryFile).filter(MemoryFile.profile_id == profile_id).order_by(MemoryFile.created_at.desc()).all()
    conversations = db.query(Conversation).filter(Conversation.profile_id == profile_id).order_by(Conversation.updated_at.desc()).all()
    total_messages = 0
    for conv in conversations:
        total_messages += db.query(Message).filter(Message.conversation_id == conv.id).count()
    completeness = compute_completeness(profile, files)
    return templates.TemplateResponse(request, "profile_detail.html", {
        "request": request, "user": user, "profile": profile, "files": files,
        "conversations": conversations, "total_messages": total_messages,
        "completeness": completeness,
    })


@router.post("/{profile_id}/update")
async def update_profile(
    request: Request,
    profile_id: str,
    name: str = Form(...),
    description: str = Form(""),
    relationship: str = Form(""),
    date_of_birth: str = Form(""),
    voice_id: str = Form(""),
    personality_traits: str = Form(""),
    favorite_phrases: str = Form(""),
    interests: str = Form(""),
    speaking_style: str = Form(""),
    writing_style: str = Form(""),
    values: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)

    profile.name = (name or "").strip() or profile.name
    profile.description = description
    profile.relationship_type = relationship
    profile.date_of_birth = date_of_birth
    profile.voice_id = voice_id
    profile.personality_traits = [t.strip() for t in personality_traits.split(",") if t.strip()] if personality_traits else []
    profile.favorite_phrases = [p.strip() for p in favorite_phrases.split("\n") if p.strip()] if favorite_phrases else []
    profile.interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else []
    profile.speaking_style = speaking_style
    profile.writing_style = writing_style
    profile.values = [v.strip() for v in values.split(",") if v.strip()] if values else []
    db.commit()

    audit(db, user.id, "update_profile", resource_type="profile", resource_id=profile_id, details=f"Updated profile: {profile.name}")
    return RedirectResponse(url=f"/profiles/{profile_id}", status_code=302)


@router.post("/{profile_id}/photo")
async def upload_profile_photo(
    request: Request,
    profile_id: str,
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)

    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in IMAGE_TYPES:
        return JSONResponse({"error": "Profile photo must be an image (PNG, JPG, JPEG, WEBP)"}, status_code=400)

    content = await photo.read()
    if len(content) > 10 * 1024 * 1024:
        return JSONResponse({"error": "Profile photo too large (max 10MB)"}, status_code=400)

    # Remove old photo file.
    if profile.photo_url:
        old_path = os.path.join(settings.UPLOAD_DIR, profile.photo_url)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    photo_filename = f"profile_photo_{profile_id}{ext}"
    with open(os.path.join(settings.UPLOAD_DIR, photo_filename), "wb") as f:
        f.write(content)
    profile.photo_url = photo_filename
    db.commit()
    audit(db, user.id, "update_profile_photo", resource_type="profile", resource_id=profile_id, details=f"Updated photo for {profile.name}")
    return JSONResponse({"success": True, "photo_url": photo_filename})


@router.get("/{profile_id}/photo")
async def get_profile_photo(request: Request, profile_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    profile = get_profile_for_user(db, profile_id, user)
    if not profile or not profile.photo_url:
        raise HTTPException(status_code=404, detail="Not found")
    file_path = os.path.join(settings.UPLOAD_DIR, profile.photo_url)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Not found")
    ext = os.path.splitext(profile.photo_url)[1].lower()
    return FileResponse(path=file_path, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"))


@router.post("/{profile_id}/delete")
async def delete_profile(request: Request, profile_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = get_profile_for_user(db, profile_id, user)
    if profile:
        for f in db.query(MemoryFile).filter(MemoryFile.profile_id == profile_id).all():
            _delete_file_from_disk(f)
        if profile.photo_url:
            old_path = os.path.join(settings.UPLOAD_DIR, profile.photo_url)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        audit(db, user.id, "delete_profile", resource_type="profile", resource_id=profile_id, details=f"Deleted profile: {profile.name}")
        db.delete(profile)
        db.commit()
    return RedirectResponse(url="/profiles", status_code=302)


@router.post("/{profile_id}/upload")
async def upload_file(
    request: Request,
    profile_id: str,
    file: UploadFile = File(...),
    caption: str = Form(""),
    memory_date: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)

    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = [e.strip() for e in settings.ALLOWED_EXTENSIONS.split(",")]
    if ext not in allowed:
        return JSONResponse({"error": f"File type {ext} not allowed. Allowed: {', '.join(allowed)}"}, status_code=400)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        return JSONResponse({"error": f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB"}, status_code=400)
    if len(content) == 0:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    with open(file_path, "wb") as f:
        f.write(content)

    memory_file = MemoryFile(
        id=file_id,
        profile_id=profile_id,
        filename=filename,
        original_name=file.filename or filename,
        file_type=ext,
        file_size=len(content),
        caption=caption.strip(),
        memory_date=(memory_date or "").strip(),
        status="uploading",
        is_processed=False,
    )
    db.add(memory_file)
    db.commit()

    audit(db, user.id, "upload_file", resource_type="file", resource_id=file_id, details=f"Uploaded {file.filename}")
    start_processing(file_id)

    return JSONResponse({
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "status": "uploading",
    })


@router.get("/{profile_id}/files/{file_id}/status")
async def file_status(request: Request, profile_id: str, file_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    file_obj = _get_owned_file(db, user, profile_id, file_id)
    details = {}
    try:
        details = json.loads(file_obj.processing_details or "{}")
    except (json.JSONDecodeError, TypeError):
        pass
    return JSONResponse({
        "file_id": file_obj.id,
        "status": file_obj.status,
        "memory_type": file_obj.memory_type,
        "chunk_count": file_obj.chunk_count,
        "word_count": file_obj.word_count,
        "transcript": file_obj.transcript or "",
        "vision_description": file_obj.vision_description or "",
        "error_message": file_obj.error_message or "",
        "processing_details": details,
    })


@router.post("/{profile_id}/files/{file_id}/delete")
async def delete_file(request: Request, profile_id: str, file_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    try:
        file_obj = _get_owned_file(db, user, profile_id, file_id)
    except HTTPException:
        return RedirectResponse(url=f"/profiles/{profile_id}", status_code=302)
    _delete_file_from_disk(file_obj)
    audit(db, user.id, "delete_file", resource_type="file", resource_id=file_id, details=f"Deleted file: {file_obj.original_name}")
    db.delete(file_obj)
    db.commit()
    return RedirectResponse(url=f"/profiles/{profile_id}", status_code=302)


@router.get("/{profile_id}/files/{file_id}/download")
async def download_file(request: Request, profile_id: str, file_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    file_obj = _get_owned_file(db, user, profile_id, file_id)
    file_path = os.path.join(settings.UPLOAD_DIR, file_obj.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(file_obj.filename)[1].lower()
    media_type = MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(path=file_path, filename=file_obj.original_name, media_type=media_type)


@router.get("/{profile_id}/files/{file_id}/view")
async def view_file(request: Request, profile_id: str, file_id: str, db: Session = Depends(get_db)):
    """Authenticated inline viewing (used for image thumbnails/previews)."""
    user = get_user_from_request(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    file_obj = _get_owned_file(db, user, profile_id, file_id)
    ext = os.path.splitext(file_obj.filename)[1].lower()
    if ext not in IMAGE_TYPES:
        raise HTTPException(status_code=404, detail="Preview not available for this file type")
    file_path = os.path.join(settings.UPLOAD_DIR, file_obj.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type=MEDIA_TYPES.get(ext, "image/jpeg"))


@router.get("/{profile_id}/timeline", response_class=HTMLResponse)
async def profile_timeline(request: Request, profile_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)
    files = db.query(MemoryFile).filter(MemoryFile.profile_id == profile_id).order_by(MemoryFile.created_at.asc()).all()

    timeline_items = []
    undated = []
    for f in files:
        label = {
            "document": "Document",
            "written": "Written Memory",
            "photograph": "Photograph",
            "audio": "Audio Recording",
            "video": "Video",
        }.get(f.memory_type, "Memory")
        title = os.path.splitext(f.original_name)[0] or f.original_name
        date_value = (f.memory_date or "").strip()
        year = date_value[:4] if len(date_value) >= 4 else ""
        item = {
            "type": "file",
            "date": date_value,
            "year": year,
            "title": title,
            "label": label,
            "id": f.id,
            "file_id": f.id,
        }
        if date_value:
            timeline_items.append(item)
        else:
            undated.append(item)

    timeline_items.sort(key=lambda x: x["year"] or "9999")

    return templates.TemplateResponse(request, "timeline.html", {
        "request": request, "user": user, "profile": profile,
        "timeline_items": timeline_items, "undated": undated,
        "date_of_birth": profile.date_of_birth,
    })


@router.get("/api/{profile_id}")
async def api_profile(request: Request, profile_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    profile = get_profile_for_user(db, profile_id, user)
    if not profile:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "relationship_type": profile.relationship_type,
        "date_of_birth": profile.date_of_birth,
        "voice_id": profile.voice_id or "",
        "personality_traits": profile.personality_traits or [],
        "favorite_phrases": profile.favorite_phrases or [],
        "interests": profile.interests or [],
        "speaking_style": profile.speaking_style or "",
        "writing_style": profile.writing_style or "",
        "values": profile.values or [],
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
    })
