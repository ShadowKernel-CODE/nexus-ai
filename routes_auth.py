import json
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db, User
from auth import hash_password, verify_password, create_access_token, get_user_from_request

router = APIRouter(prefix="/auth", tags=["auth"])


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
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "user": None,
            "error": "Invalid email or password",
        })

    token = create_access_token({"sub": user.id})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie("session_token", token, httponly=True, samesite="lax", max_age=86400)
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

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(request, "register.html", {
            "request": request,
            "user": None,
            "error": "An account with this email already exists",
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
    response.set_cookie("session_token", token, httponly=True, samesite="lax", max_age=86400)
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/auth/login", status_code=302)
    response.delete_cookie("session_token")
    return response
