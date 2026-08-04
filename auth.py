from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from fastapi import Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from config import settings
from database import get_db, User, MemoryProfile, MemoryFile, Conversation

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


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
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def require_auth_page(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """For page routes - returns user or None (page handles redirect)."""
    return get_user_from_request(request, db)


def require_admin(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_user_from_request(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Forbidden")
    return user


def get_profile_for_user(db: Session, profile_id: str, user: User):
    return db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id,
        MemoryProfile.user_id == user.id,
    ).first()


def get_file_for_profile(db: Session, file_id: str, profile_id: str):
    return db.query(MemoryFile).filter(
        MemoryFile.id == file_id,
        MemoryFile.profile_id == profile_id,
    ).first()


def get_conversation_for_user(db: Session, conversation_id: str, user: User):
    return db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
