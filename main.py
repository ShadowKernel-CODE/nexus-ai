from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import os

from config import settings
from database import init_db, SessionLocal, User

app = FastAPI(title=settings.APP_NAME, description=settings.APP_DESCRIPTION)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

from routes_auth import router as auth_router
from routes_profiles import router as profiles_router
from routes_chat import router as chat_router
from routes_search import router as search_router
from routes_admin import router as admin_router
from routes_voice import router as voice_router
from routes_library import router as library_router

app.include_router(auth_router)
app.include_router(profiles_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(admin_router)
app.include_router(voice_router)
app.include_router(library_router)


@app.on_event("startup")
async def startup():
    init_db()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    from memory_processing import recover_stale_processing
    recover_stale_processing()
    try:
        from seed import seed
        seed()
    except Exception:
        import traceback
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    from auth import get_user_from_request
    db = SessionLocal()
    try:
        user = get_user_from_request(request, db)
        if user:
            return RedirectResponse(url="/dashboard", status_code=302)
        return RedirectResponse(url="/auth/login", status_code=302)
    finally:
        db.close()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    from auth import get_user_from_request
    from database import MemoryProfile, Conversation, MemoryFile
    from completeness import compute_completeness
    db = SessionLocal()
    try:
        user = get_user_from_request(request, db)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)

        profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
        total_conversations = db.query(Conversation).filter(Conversation.user_id == user.id).count()

        profile_ids = [p.id for p in profiles]
        files_by_profile = {}
        companion_data = []
        for p in profiles:
            files = db.query(MemoryFile).filter(MemoryFile.profile_id == p.id).all()
            files_by_profile[p.id] = files
            recent = max((f.created_at for f in files), default=None)
            companion_data.append({
                "profile": p,
                "file_count": len(files),
                "recent_memory": recent,
                "completeness": compute_completeness(p, files),
            })

        recent_files = (
            db.query(MemoryFile)
            .join(MemoryProfile, MemoryProfile.id == MemoryFile.profile_id)
            .filter(MemoryProfile.user_id == user.id)
            .order_by(MemoryFile.created_at.desc())
            .limit(8)
            .all()
        )
        file_profile_names = {}
        for f in recent_files:
            p = db.query(MemoryProfile).filter(MemoryProfile.id == f.profile_id).first()
            if p:
                file_profile_names[f.id] = p.name

        recent_conversations = db.query(Conversation).filter(
            Conversation.user_id == user.id
        ).order_by(Conversation.updated_at.desc()).limit(5).all()

        recent_conv_profiles = {}
        for rc in recent_conversations:
            p = db.query(MemoryProfile).filter(MemoryProfile.id == rc.profile_id).first()
            if p:
                recent_conv_profiles[rc.id] = p.name

        return templates.TemplateResponse(request, "dashboard.html", {
            "request": request, "user": user, "profiles": profiles,
            "companion_data": companion_data,
            "recent_files": recent_files,
            "file_profile_names": file_profile_names,
            "total_conversations": total_conversations,
            "recent_conversations": recent_conversations,
            "recent_conv_profiles": recent_conv_profiles,
        })
    finally:
        db.close()


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    from auth import get_user_from_request
    db = SessionLocal()
    try:
        user = get_user_from_request(request, db)
        return templates.TemplateResponse(request, "about.html", {"request": request, "user": user})
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
