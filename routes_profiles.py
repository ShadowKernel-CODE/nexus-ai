from fastapi import APIRouter, Request, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from collections import defaultdict
from datetime import datetime
import os
import uuid
import json

from database import get_db, User, MemoryProfile, MemoryFile, MemoryEmbedding, Conversation, Message
from auth import get_user_from_request
from text_extractor import extract_text_from_file
from rag import generate_embedding
from config import settings

router = APIRouter(prefix="/profiles", tags=["profiles"])


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
    date_of_death: str = Form(""),
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

    profile = MemoryProfile(
        user_id=user.id,
        name=name,
        description=description,
        relationship_type=relationship,
        date_of_birth=date_of_birth,
        date_of_death=date_of_death,
        personality_traits=[t.strip() for t in personality_traits.split(",") if t.strip()] if personality_traits else [],
        favorite_phrases=[p.strip() for p in favorite_phrases.split("\n") if p.strip()] if favorite_phrases else [],
        interests=[i.strip() for i in interests.split(",") if i.strip()] if interests else [],
        speaking_style=speaking_style,
        writing_style=writing_style,
        values=[v.strip() for v in values.split(",") if v.strip()] if values else [],
    )
    db.add(profile)
    db.commit()

    from database import AuditLog
    log = AuditLog(user_id=user.id, action="create_profile", resource_type="profile", resource_id=profile.id, details=f"Created profile: {name}")
    db.add(log)
    db.commit()

    return RedirectResponse(url=f"/profiles/{profile.id}", status_code=302)


@router.get("/{profile_id}", response_class=HTMLResponse)
async def profile_detail(request: Request, profile_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id,
        MemoryProfile.user_id == user.id,
    ).first()
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)
    files = db.query(MemoryFile).filter(MemoryFile.profile_id == profile_id).order_by(MemoryFile.created_at.desc()).all()
    conversations = db.query(Conversation).filter(Conversation.profile_id == profile_id).order_by(Conversation.updated_at.desc()).all()
    total_messages = 0
    for conv in conversations:
        total_messages += db.query(Message).filter(Message.conversation_id == conv.id).count()
    return templates.TemplateResponse(request, "profile_detail.html", {
        "request": request, "user": user, "profile": profile, "files": files,
        "conversations": conversations, "total_messages": total_messages,
    })


@router.post("/{profile_id}/update")
async def update_profile(
    request: Request,
    profile_id: str,
    name: str = Form(...),
    description: str = Form(""),
    relationship: str = Form(""),
    date_of_birth: str = Form(""),
    date_of_death: str = Form(""),
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
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)

    profile.name = name
    profile.description = description
    profile.relationship_type = relationship
    profile.date_of_birth = date_of_birth
    profile.date_of_death = date_of_death
    profile.personality_traits = [t.strip() for t in personality_traits.split(",") if t.strip()] if personality_traits else []
    profile.favorite_phrases = [p.strip() for p in favorite_phrases.split("\n") if p.strip()] if favorite_phrases else []
    profile.interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else []
    profile.speaking_style = speaking_style
    profile.writing_style = writing_style
    profile.values = [v.strip() for v in values.split(",") if v.strip()] if values else []
    db.commit()

    return RedirectResponse(url=f"/profiles/{profile_id}", status_code=302)


@router.post("/{profile_id}/delete")
async def delete_profile(request: Request, profile_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if profile:
        db.delete(profile)
        db.commit()
    return RedirectResponse(url="/profiles", status_code=302)


@router.post("/{profile_id}/upload")
async def upload_file(
    request: Request,
    profile_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return JSONResponse({"error": "Profile not found"}, status_code=404)

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = [e.strip() for e in settings.ALLOWED_EXTENSIONS.split(",")]
    if ext not in allowed:
        return JSONResponse({"error": f"File type {ext} not allowed. Allowed: {', '.join(allowed)}"}, status_code=400)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        return JSONResponse({"error": f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB"}, status_code=400)

    with open(file_path, "wb") as f:
        f.write(content)

    extracted_text, chunks = extract_text_from_file(file_path, ext)

    memory_file = MemoryFile(
        id=file_id,
        profile_id=profile_id,
        filename=filename,
        original_name=file.filename,
        file_type=ext,
        file_size=len(content),
        extracted_text=extracted_text,
        text_chunks=json.dumps(chunks),
    )
    db.add(memory_file)
    db.commit()

    for i, chunk in enumerate(chunks):
        embedding_vec = generate_embedding(chunk)
        emb = MemoryEmbedding(
            profile_id=profile_id,
            file_id=file_id,
            content=chunk,
            embedding=json.dumps(embedding_vec) if embedding_vec else "[]",
            chunk_index=i,
        )
        db.add(emb)
    db.commit()

    from database import AuditLog
    log = AuditLog(user_id=user.id, action="upload_file", resource_type="file", resource_id=file_id, details=f"Uploaded {file.filename} ({len(chunks)} chunks)")
    db.add(log)
    db.commit()

    return JSONResponse({
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "text_preview": extracted_text[:200] if extracted_text else "",
    })


@router.post("/{profile_id}/files/{file_id}/delete")
async def delete_file(request: Request, profile_id: str, file_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    file = db.query(MemoryFile).filter(
        MemoryFile.id == file_id, MemoryFile.profile_id == profile_id
    ).first()
    if file:
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(file)
        db.commit()
    return RedirectResponse(url=f"/profiles/{profile_id}", status_code=302)


@router.get("/{profile_id}/timeline", response_class=HTMLResponse)
async def profile_timeline(request: Request, profile_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return RedirectResponse(url="/profiles", status_code=302)
    files = db.query(MemoryFile).filter(MemoryFile.profile_id == profile_id).order_by(MemoryFile.created_at.asc()).all()
    conversations = db.query(Conversation).filter(Conversation.profile_id == profile_id).order_by(Conversation.created_at.asc()).all()

    timeline_items = []
    for f in files:
        timeline_items.append({
            "type": "file",
            "date": f.created_at,
            "title": f.original_name,
            "detail": f"{f.file_type} file, {(f.file_size / 1024)|round(1)}KB",
            "id": f.id,
        })
    for c in conversations:
        timeline_items.append({
            "type": "conversation",
            "date": c.created_at,
            "title": c.title,
            "detail": f"Started conversation",
            "id": c.id,
        })
    timeline_items.sort(key=lambda x: x["date"] or datetime.min, reverse=True)

    monthly = defaultdict(list)
    for item in timeline_items:
        if item["date"]:
            key = item["date"].strftime("%B %Y")
            monthly[key].append(item)

    return templates.TemplateResponse(request, "timeline.html", {
        "request": request, "user": user, "profile": profile,
        "monthly": dict(monthly),
    })


@router.get("/api/{profile_id}")
async def api_profile(request: Request, profile_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "relationship_type": profile.relationship_type,
        "date_of_birth": profile.date_of_birth,
        "date_of_death": profile.date_of_death,
        "personality_traits": profile.personality_traits or [],
        "favorite_phrases": profile.favorite_phrases or [],
        "interests": profile.interests or [],
        "speaking_style": profile.speaking_style or "",
        "writing_style": profile.writing_style or "",
        "values": profile.values or [],
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
    })
