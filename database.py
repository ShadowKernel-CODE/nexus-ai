from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
import uuid

from config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def gen_id():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_id)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    profiles = relationship("MemoryProfile", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")


class MemoryProfile(Base):
    __tablename__ = "memory_profiles"
    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    relationship_type = Column(String, default="")
    date_of_birth = Column(String, default="")
    date_of_death = Column(String, default="")
    voice_id = Column(String, default="")
    photo_url = Column(String, default="")
    personality_traits = Column(JSON, default=list)
    favorite_phrases = Column(JSON, default=list)
    interests = Column(JSON, default=list)
    speaking_style = Column(Text, default="")
    writing_style = Column(Text, default="")
    values = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="profiles")
    files = relationship("MemoryFile", back_populates="profile", cascade="all, delete-orphan")
    embeddings = relationship("MemoryEmbedding", back_populates="profile", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="profile", cascade="all, delete-orphan")


class MemoryFile(Base):
    __tablename__ = "memory_files"
    id = Column(String, primary_key=True, default=gen_id)
    profile_id = Column(String, ForeignKey("memory_profiles.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    extracted_text = Column(Text, default="")
    text_chunks = Column(Text, default="[]")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    status = Column(String, default="uploading")  # uploading|extracting|transcribing|analyzing|indexing|ready|failed
    memory_type = Column(String, default="")       # document|written|photograph|audio|video
    caption = Column(Text, default="")             # user-provided context
    memory_date = Column(String, default="")       # known date, e.g. 1998-07-12 ("" = unknown)
    transcript = Column(Text, default="")          # audio/video transcript
    vision_description = Column(Text, default="")  # image description from vision model
    word_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, default="")
    processing_details = Column(Text, default="{}")
    is_processed = Column(Boolean, default=False)

    profile = relationship("MemoryProfile", back_populates="files")
    embeddings = relationship("MemoryEmbedding", back_populates="file", cascade="all, delete-orphan")


class MemoryEmbedding(Base):
    __tablename__ = "memory_embeddings"
    id = Column(String, primary_key=True, default=gen_id)
    profile_id = Column(String, ForeignKey("memory_profiles.id"), nullable=False)
    file_id = Column(String, ForeignKey("memory_files.id"), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, default="[]")
    chunk_index = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    profile = relationship("MemoryProfile", back_populates="embeddings")
    file = relationship("MemoryFile", back_populates="embeddings")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    profile_id = Column(String, ForeignKey("memory_profiles.id"), nullable=False)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="conversations")
    profile = relationship("MemoryProfile", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=gen_id)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, default="[]")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, default="")
    resource_id = Column(String, default="")
    details = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="audit_logs")


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate()


def migrate():
    from sqlalchemy import inspect, text
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            profile_columns = [c["name"] for c in inspector.get_columns("memory_profiles")]
            if "voice_id" not in profile_columns:
                conn.execute(text("ALTER TABLE memory_profiles ADD COLUMN voice_id VARCHAR DEFAULT ''"))
                conn.commit()

            user_columns = [c["name"] for c in inspector.get_columns("users")]
            if "is_admin" not in user_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0"))
                conn.commit()

            file_columns = [c["name"] for c in inspector.get_columns("memory_files")]
            file_additions = {
                "status": "VARCHAR DEFAULT 'ready'",
                "memory_type": "VARCHAR DEFAULT ''",
                "caption": "TEXT DEFAULT ''",
                "memory_date": "VARCHAR DEFAULT ''",
                "transcript": "TEXT DEFAULT ''",
                "vision_description": "TEXT DEFAULT ''",
                "word_count": "INTEGER DEFAULT 0",
                "chunk_count": "INTEGER DEFAULT 0",
                "error_message": "TEXT DEFAULT ''",
                "processing_details": "TEXT DEFAULT '{}'",
                "is_processed": "BOOLEAN DEFAULT 0",
            }
            for col, coldef in file_additions.items():
                if col not in file_columns:
                    conn.execute(text(f"ALTER TABLE memory_files ADD COLUMN {col} {coldef}"))
                    conn.commit()
    except Exception as e:
        print(f"Migration warning: {e}")
