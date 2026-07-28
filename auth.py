from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from config import settings
from database import get_db, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=settings.SESSION_EXPIRY_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def get_user_from_request(request: Request, db: Session) -> Optional[User]:
    token = request.cookies.get("session_token")
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def require_auth(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_user_from_request(request, db)
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/auth/login"})
    return user


def require_auth_page(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """For page routes - returns user or None (page handles redirect)."""
    return get_user_from_request(request, db)
