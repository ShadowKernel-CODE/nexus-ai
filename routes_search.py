import json
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db, User, MemoryProfile, MemoryFile, MemoryEmbedding, Conversation, Message, AuditLog
from auth import get_user_from_request
from rag import search_similar_memories

router = APIRouter(tags=["search"])


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    return templates.TemplateResponse("search.html", {
        "request": request, "user": user, "profiles": profiles,
        "results": [], "query": "",
    })


@router.post("/search")
async def search_execute(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    form = await request.form()
    query = form.get("query", "")
    profile_id = form.get("profile_id", "")

    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    results = []

    if query and profile_id:
        memories = search_similar_memories(db, profile_id, query, limit=20)
        results = memories
    elif query:
        for p in profiles:
            memories = search_similar_memories(db, p.id, query, limit=5)
            for m in memories:
                m["profile_name"] = p.name
                m["profile_id"] = p.id
            results.extend(memories)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return templates.TemplateResponse("search.html", {
        "request": request, "user": user, "profiles": profiles,
        "results": results, "query": query, "selected_profile": profile_id,
    })


@router.get("/api/search")
async def api_search(request: Request, q: str = "", profile_id: str = "", db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not q:
        return JSONResponse({"results": []})

    if profile_id:
        results = search_similar_memories(db, profile_id, q, limit=20)
    else:
        profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
        results = []
        for p in profiles:
            memories = search_similar_memories(db, p.id, q, limit=5)
            for m in memories:
                m["profile_name"] = p.name
                m["profile_id"] = p.id
            results.extend(memories)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return JSONResponse({"results": results, "query": q})
