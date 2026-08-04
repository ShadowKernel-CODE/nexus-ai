import json
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db, User, AuditLog
from auth import hash_password, verify_password, create_access_token, get_user_from_request
from config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

MIN_PASSWORD_LENGTH = 8


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def audit(db: Session, user_id, action, resource_type="", resource_id="", details=""):
    db.add(AuditLog(user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, details=details))
    db.commit()


def set_session_cookie(response: RedirectResponse, token: str) -> None:
    response.set_cookie(
        "session_token",
        token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.SESSION_EXPIRY_HOURS * 3600,
        path='/',
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"request": request, "user": None, "error": None})


@router.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = db.query(User).filter(User.email == normalize_email(email)).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "user": None,
            "error": "Invalid email or password",
        })

    token = create_access_token({"sub": user.id})
    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, token)
    audit(db, user.id, "login", resource_type="auth")
    return response


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"request": request, "user": None, "error": None})


@router.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")

    name = (name or "").strip()
    email = normalize_email(email)

    error = None
    if not name or not email or "@" not in email:
        error = "Please provide a valid name and email address"
    elif len(password) < MIN_PASSWORD_LENGTH:
        error = f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    else:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            error = "An account with this email already exists"

    if error:
        return templates.TemplateResponse(request, "register.html", {
            "request": request,
            "user": None,
            "error": error,
        })

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id})
    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, token)
    audit(db, user.id, "register", resource_type="auth")
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/auth/login", status_code=302)
    response.delete_cookie("session_token", path="/")
    response.set_cookie("session_token", "", max_age=0, path="/", httponly=True, samesite="lax", secure=settings.cookie_secure)
    return response
