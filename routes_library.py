"""Memory Library: a visual, filterable view of every preserved memory
across all of the authenticated user's profiles.

Modality filters: All / Stories (written) / Documents / Photos / Audio / Video.
"""
import os
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db, MemoryProfile, MemoryFile
from auth import get_user_from_request

router = APIRouter(prefix="/library", tags=["library"])

FILTER_LABELS = {
    "": "All",
    "written": "Stories",
    "document": "Documents",
    "photograph": "Photos",
    "audio": "Audio",
    "video": "Video",
}


def _modality_label(memory_type: str) -> str:
    return {
        "written": "Written Memory",
        "document": "Document",
        "photograph": "Photograph",
        "audio": "Audio Recording",
        "video": "Video",
    }.get(memory_type or "", "Memory")


def _excerpt(file_obj: MemoryFile, limit: int = 150) -> str:
    text = (
        file_obj.transcript
        or file_obj.vision_description
        or file_obj.extracted_text
        or file_obj.caption
        or ""
    ).strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


@router.get("", response_class=HTMLResponse)
async def library_page(request: Request, type: str = "", db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    selected = type if type in FILTER_LABELS else ""

    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    profile_ids = [p.id for p in profiles]
    profile_names = {p.id: p.name for p in profiles}

    query = (
        db.query(MemoryFile)
        .filter(MemoryFile.profile_id.in_(profile_ids))
        .order_by(MemoryFile.created_at.desc())
    )
    if selected:
        query = query.filter(MemoryFile.memory_type == selected)
    files = query.all()

    counts = {key: 0 for key in FILTER_LABELS}
    for f in db.query(MemoryFile).filter(MemoryFile.profile_id.in_(profile_ids)).all():
        key = f.memory_type if f.memory_type in counts else ""
        counts[key] = counts.get(key, 0) + 1

    items = []
    for f in files:
        items.append({
            "id": f.id,
            "profile_id": f.profile_id,
            "profile_name": profile_names.get(f.profile_id, ""),
            "original_name": f.original_name or f.filename,
            "title": os.path.splitext(f.original_name or f.filename)[0],
            "memory_type": f.memory_type or "",
            "modality_label": _modality_label(f.memory_type),
            "status": f.status or "uploading",
            "memory_date": f.memory_date or "",
            "caption": f.caption or "",
            "excerpt": _excerpt(f),
            "error_message": f.error_message or "",
            "is_media": f.memory_type in ("photograph", "audio", "video"),
        })

    return templates.TemplateResponse(request, "library.html", {
        "request": request,
        "user": user,
        "items": items,
        "selected": selected,
        "filter_labels": FILTER_LABELS,
        "counts": counts,
        "total": sum(counts.values()),
    })
