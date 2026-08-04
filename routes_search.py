import json
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db, User, MemoryProfile, MemoryFile, MemoryEmbedding, Conversation, Message, AuditLog
from auth import get_user_from_request, get_profile_for_user
from rag import search_similar_memories

router = APIRouter(tags=["search"])


def _owned_results(db: Session, user: User, query: str, profile_id: str, per_profile: int = 5, total_limit: int = 20):
    """Search across the user's profiles. profile_id (if given) must be owned by the user."""
    results = []
    if profile_id:
        profile = get_profile_for_user(db, profile_id, user)
        if not profile:
            return []
        memories = search_similar_memories(db, profile_id, query, limit=total_limit)
        for m in memories:
            m["profile_name"] = profile.name
            m["profile_id"] = profile.id
        results.extend(memories)
    else:
        profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
        for p in profiles:
            memories = search_similar_memories(db, p.id, query, limit=per_profile)
            for m in memories:
                m["profile_name"] = p.name
                m["profile_id"] = p.id
            results.extend(memories)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:total_limit]


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    return templates.TemplateResponse(request, "search.html", {
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
    if query:
        results = _owned_results(db, user, query, profile_id)

    return templates.TemplateResponse(request, "search.html", {
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

    results = _owned_results(db, user, q, profile_id)
    return JSONResponse({"results": results, "query": q})
