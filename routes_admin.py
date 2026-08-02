from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, User, MemoryProfile, MemoryFile, Conversation, Message, AuditLog
from auth import get_user_from_request

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", response_class=HTMLResponse)
async def admin_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    stats = {
        "total_users": db.query(User).count(),
        "total_profiles": db.query(MemoryProfile).count(),
        "total_files": db.query(MemoryFile).count(),
        "total_conversations": db.query(Conversation).count(),
        "total_messages": db.query(Message).count(),
    }

    recent_logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(50).all()
    users = db.query(User).all()
    profiles = db.query(MemoryProfile).all()

    return templates.TemplateResponse(request, "admin.html", {
        "request": request, "user": user, "stats": stats,
        "recent_logs": recent_logs, "users": users, "profiles": profiles,
    })


@router.get("/api/stats")
async def admin_stats(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    return JSONResponse({
        "total_users": db.query(User).count(),
        "total_profiles": db.query(MemoryProfile).count(),
        "total_files": db.query(MemoryFile).count(),
        "total_conversations": db.query(Conversation).count(),
        "total_messages": db.query(Message).count(),
        "total_embeddings": db.query(MemoryEmbedding).count() if hasattr(db.query(MemoryProfile), 'count') else 0,
    })


@router.get("/export")
async def export_data(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from database import MemoryEmbedding
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    export = []
    for p in profiles:
        files = db.query(MemoryFile).filter(MemoryFile.profile_id == p.id).all()
        conversations = db.query(Conversation).filter(Conversation.profile_id == p.id).all()
        conv_data = []
        for c in conversations:
            msgs = db.query(Message).filter(Message.conversation_id == c.id).order_by(Message.created_at).all()
            conv_data.append({
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "messages": [{"role": m.role, "content": m.content} for m in msgs],
            })
        export.append({
            "profile": {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "relationship": p.relationship_type,
                "date_of_birth": p.date_of_birth,
                "date_of_death": p.date_of_death,
            },
            "files": [{"id": f.id, "filename": f.original_name, "file_type": f.file_type, "file_size": f.file_size} for f in files],
            "conversations": conv_data,
        })

    return JSONResponse(export)


@router.get("/logs")
async def admin_logs(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(200).all()
    return JSONResponse([{
        "id": l.id, "action": l.action, "resource_type": l.resource_type,
        "resource_id": l.resource_id, "details": l.details,
        "created_at": l.created_at.isoformat() if l.created_at else None,
    } for l in logs])
