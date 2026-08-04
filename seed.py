"""
Seed script for MemoryBot.
Creates demo users and a polished multimodal memory profile for Margaret.
Run: python seed.py

Idempotent: safe to run on every startup. Never creates privileged accounts
unless ADMIN_EMAIL/ADMIN_PASSWORD are explicitly configured.
"""
import sys
sys.path.insert(0, ".")

import json
import math
import os
import wave

from database import init_db, SessionLocal, User, MemoryProfile, MemoryFile, MemoryEmbedding, AuditLog
from auth import hash_password
from config import settings
from rag import generate_embedding
from text_extractor import chunk_text

DEMO_EMAIL = "demo@memorybot.com"
DEMO_PASSWORD = "demo123"


def _audit(db, user_id, action, resource_type="", resource_id="", details=""):
    db.add(AuditLog(user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, details=details))
    db.commit()


def _safe_embed(db, profile_id, file_id, chunks):
    for i, chunk in enumerate(chunks):
        emb_vec = generate_embedding(chunk)
        db.add(MemoryEmbedding(
            profile_id=profile_id,
            file_id=file_id,
            content=chunk,
            embedding=json.dumps(emb_vec) if emb_vec else "[]",
            chunk_index=i,
        ))


def _write_demo_file(filename, content: bytes):
    path = os.path.join(settings.UPLOAD_DIR, filename)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    return path


def _ensure_written_memory(db, profile, title, content, memory_date=""):
    """Idempotently create a written memory (txt) for the demo profile."""
    original = f"{title}.txt"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        if not existing.memory_type:
            existing.memory_type = "written"
            existing.status = "ready"
            existing.is_processed = True
            existing.chunk_count = len(json.loads(existing.text_chunks or "[]") or [])
        if memory_date:
            existing.memory_date = memory_date
        db.commit()
        return existing

    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"sample_{safe}.txt"
    _write_demo_file(filename, content.encode("utf-8"))

    chunks = chunk_text(content)
    f = MemoryFile(
        profile_id=profile.id,
        filename=filename,
        original_name=original,
        file_type=".txt",
        file_size=len(content.encode("utf-8")),
        extracted_text=content,
        text_chunks=json.dumps(chunks),
        memory_type="written",
        memory_date=memory_date,
        status="ready",
        is_processed=True,
        word_count=len(content.split()),
        chunk_count=len(chunks),
        processing_details=json.dumps({"seed": True}),
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    _safe_embed(db, profile.id, f.id, chunks)
    db.commit()
    return f


def _ensure_document_memory(db, profile, title, content, memory_date=""):
    """Create a real .docx document memory."""
    original = f"{title}.docx"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        return existing

    from docx import Document
    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"demo_{safe}.docx"
    doc = Document()
    for para in content.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    tmp = os.path.join(settings.UPLOAD_DIR, filename)
    doc.save(tmp)
    size = os.path.getsize(tmp)

    chunks = chunk_text(content)
    f = MemoryFile(
        profile_id=profile.id,
        filename=filename,
        original_name=original,
        file_type=".docx",
        file_size=size,
        extracted_text=content,
        text_chunks=json.dumps(chunks),
        memory_type="document",
        memory_date=memory_date,
        status="ready",
        is_processed=True,
        word_count=len(content.split()),
        chunk_count=len(chunks),
        processing_details=json.dumps({"seed": True}),
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    _safe_embed(db, profile.id, f.id, chunks)
    db.commit()
    return f


def _make_demo_garden_image(path: str):
    """Draw a simple garden scene so the demo photograph is viewable."""
    from PIL import Image, ImageDraw
    width, height = 640, 480
    img = Image.new("RGB", (width, height), "#8ecae6")
    d = ImageDraw.Draw(img)
    # sky gradient feel + sun
    d.rectangle([0, 0, width, int(height * 0.5)], fill="#a8d8ea")
    d.ellipse([width - 140, 40, width - 60, 120], fill="#f7d774")
    # hills
    d.rectangle([0, int(height * 0.5), width, height], fill="#7fb069")
    # wooden fence
    for x in range(20, width, 70):
        d.rectangle([x, int(height * 0.42), x + 14, int(height * 0.62)], fill="#c49a6c")
        d.rectangle([x, int(height * 0.42), x + 70, int(height * 0.42) + 8], fill="#b98a5e")
    # flower beds with roses
    for row, (fx, fy) in enumerate([(60, 340), (200, 360), (340, 345), (480, 355)]):
        d.rectangle([fx - 30, fy, fx + 30, height], fill="#8d6e63")
        for ox in (-14, 0, 14):
            d.ellipse([fx + ox - 8, fy - 26, fx + ox + 8, fy - 10], fill="#b23b3b" if row % 2 else "#d94f6b")
            d.line([fx + ox, fy - 10, fx + ox, fy], fill="#3e7c3e", width=3)
    img.save(path, "JPEG", quality=85)


def _make_demo_audio(path: str):
    """Write a short soft-tone WAV so the demo audio memory is playable."""
    framerate = 8000
    duration = 4
    freq = 220.0
    frames = int(framerate * duration)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        data = bytearray()
        for i in range(frames):
            sample = int(32767 * 0.25 * math.sin(2 * math.pi * freq * i / framerate))
            data += sample.to_bytes(2, byteorder="little", signed=True)
        w.writeframes(bytes(data))


def _ensure_photo_memory(db, profile, title, description, memory_date=""):
    original = f"{title}.jpg"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        return existing

    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"demo_{safe}.jpg"
    path = os.path.join(settings.UPLOAD_DIR, filename)
    _make_demo_garden_image(path)

    content = f"[Photograph: {original}]\n\n{description}"
    chunks = chunk_text(content)
    f = MemoryFile(
        profile_id=profile.id,
        filename=filename,
        original_name=original,
        file_type=".jpg",
        file_size=os.path.getsize(path),
        extracted_text=content,
        text_chunks=json.dumps(chunks),
        vision_description=description,
        caption="Grandma in her garden, summer 1998.",
        memory_type="photograph",
        memory_date=memory_date,
        status="ready",
        is_processed=True,
        word_count=len(content.split()),
        chunk_count=len(chunks),
        processing_details=json.dumps({"seed": True, "vision": True}),
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    _safe_embed(db, profile.id, f.id, chunks)
    db.commit()
    return f


def _ensure_audio_memory(db, profile, title, transcript, memory_date=""):
    original = f"{title}.wav"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        return existing

    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"demo_{safe}.wav"
    path = os.path.join(settings.UPLOAD_DIR, filename)
    _make_demo_audio(path)

    chunks = chunk_text(transcript)
    f = MemoryFile(
        profile_id=profile.id,
        filename=filename,
        original_name=original,
        file_type=".wav",
        file_size=os.path.getsize(path),
        extracted_text=transcript,
        text_chunks=json.dumps(chunks),
        transcript=transcript,
        caption="Recorded interview.",
        memory_type="audio",
        memory_date=memory_date,
        status="ready",
        is_processed=True,
        word_count=len(transcript.split()),
        chunk_count=len(chunks),
        processing_details=json.dumps({"seed": True, "transcribed": True}),
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    _safe_embed(db, profile.id, f.id, chunks)
    db.commit()
    return f


def _ensure_video_memory(db, profile, title, transcript, memory_date=""):
    original = f"{title}.mp4"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        return existing

    import subprocess
    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"demo_{safe}.mp4"
    photo = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.memory_type == "photograph"
    ).first()
    audio = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.memory_type == "audio"
    ).first()
    image_path = os.path.join(settings.UPLOAD_DIR, photo.filename) if photo else None
    audio_path = os.path.join(settings.UPLOAD_DIR, audio.filename) if audio else None
    if not image_path or not audio_path or not os.path.exists(image_path) or not os.path.exists(audio_path):
        return None
    out_path = os.path.join(settings.UPLOAD_DIR, filename)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", image_path, "-i", audio_path,
             "-shortest", "-pix_fmt", "yuv420p", out_path],
            capture_output=True, timeout=120, check=True,
        )
    except Exception as e:
        print(f"  (skipping demo video: {e})")
        return None
    if not os.path.exists(out_path):
        return None

    chunks = chunk_text(transcript)
    f = MemoryFile(
        profile_id=profile.id,
        filename=filename,
        original_name=original,
        file_type=".mp4",
        file_size=os.path.getsize(out_path),
        extracted_text=transcript,
        text_chunks=json.dumps(chunks),
        transcript=transcript,
        caption="Family reunion footage.",
        memory_type="video",
        memory_date=memory_date,
        status="ready",
        is_processed=True,
        word_count=len(transcript.split()),
        chunk_count=len(chunks),
        processing_details=json.dumps({"seed": True, "transcribed": True, "frames_analyzed": 0}),
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    _safe_embed(db, profile.id, f.id, chunks)
    db.commit()
    return f


def seed():
    init_db()
    db = SessionLocal()

    # Remove the legacy predictable-credential account created by old seeds.
    legacy = db.query(User).filter(User.email == "admin@memorybot.com").first()
    if legacy:
        db.delete(legacy)
        db.commit()
        print("Removed legacy admin@memorybot.com account (predictable credentials)")

    # Admin account is created ONLY when explicitly configured.
    admin_email = getattr(settings, 'ADMIN_EMAIL', '').strip().lower()
    admin_password = getattr(settings, 'ADMIN_PASSWORD', '')
    if admin_email and admin_password:
        admin = db.query(User).filter(User.email == admin_email).first()
        if not admin:
            admin = User(
                name="Administrator",
                email=admin_email,
                password_hash=hash_password(admin_password),
                is_admin=True,
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
            print(f"Created admin user: {admin_email}")
        elif not admin.is_admin:
            admin.is_admin = True
            db.commit()
    else:
        print("Admin account not created automatically. Set ADMIN_EMAIL and ADMIN_PASSWORD to seed one.")

    demo_user = db.query(User).filter(User.email == DEMO_EMAIL).first()
    if not demo_user:
        demo_user = User(
            name="Demo User",
            email=DEMO_EMAIL,
            password_hash=hash_password(DEMO_PASSWORD),
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        print(f"Created demo user: {DEMO_EMAIL} / {DEMO_PASSWORD}")

    profile = db.query(MemoryProfile).filter(
        MemoryProfile.user_id == demo_user.id,
        MemoryProfile.name == "Margaret Johnson",
    ).first()
    if not profile:
        profile = MemoryProfile(
            user_id=demo_user.id,
            name="Margaret Johnson",
            description="My beloved grandmother who lived a full and inspiring life. She was known for her kindness, her incredible cooking, and the stories she would tell about growing up during the Great Depression.",
            relationship_type="Grandmother",
            date_of_birth="1925-03-15",
            voice_id="EXAVITQu4vr4xnSDxMaL",
            personality_traits=["Kind", "Patient", "Strong", "Witty", "Generous", "Resilient"],
            favorite_phrases=["Every storm runs out of rain", "You catch more flies with honey than vinegar", "A family that eats together stays together"],
            interests=["Gardening", "Reading", "Cooking", "Church activities", "Storytelling"],
            speaking_style="Warm and gentle, with a slight Southern drawl. Often used idioms and proverbs.",
            writing_style="Never wrote much, but her letters were heartfelt and full of wisdom.",
            values=["Family", "Faith", "Community", "Hard work", "Kindness", "Generosity"],
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        print("Created memory profile: Margaret Johnson")

    sample_memories = [
        {
            "title": "Childhood during the Depression",
            "date": "1933-06-14",
            "content": "Margaret was born in 1925 in a small farming town in Ohio. During the Great Depression, her family lost their farm and had to move to the city. She often told stories about how her mother would stretch a single chicken into meals that lasted three days. Despite the hardships, she always spoke of this time with a sense of resilience and community spirit. Her neighbors would share what little they had, and she learned the value of generosity from watching her parents help others even when they themselves had very little.",
        },
        {
            "title": "Meeting Grandpa Robert",
            "date": "1946-05-20",
            "content": "Margaret met Robert Johnson at a church social in 1946, just after he returned from serving in World War II. She said he was the most handsome man she had ever seen, with his military uniform and shy smile. Their first date was at the local diner where he bought her a strawberry milkshake. They married in June 1948 and were together for 52 years until Robert passed away in 2000. She often said their secret to a happy marriage was never going to bed angry and always finding something to laugh about.",
        },
        {
            "title": "Her Famous Apple Pie",
            "date": "1965-11-20",
            "content": "Margaret's apple pie was legendary in the family. The recipe came from her own grandmother and used a secret blend of cinnamon, nutmeg, and a pinch of cardamom. She would pick apples from the tree in her backyard every autumn and spend the whole day baking. The whole house would smell of cinnamon and butter. Every Thanksgiving, she would bake five pies - one for each of her children's families. She never wrote down the recipe, saying it was all in her hands and heart. After she passed, the family tried to recreate it but could never quite get it right.",
        },
        {
            "title": "The Garden",
            "date": "1998-07-12",
            "content": "Margaret had the most beautiful garden. She grew roses, tulips, and her famous sunflowers that towered over the fence. Every spring she would plan her garden layout like a general planning a campaign. She loved her tomatoes and would can hundreds of jars every summer to last through winter. The neighborhood children would come over to pick fresh vegetables and she would teach them the names of each plant. Her garden was her pride and joy, and she always said it kept her young.",
        },
        {
            "title": "Stories of the War Years",
            "date": "1944-09-03",
            "content": "Although Margaret herself did not serve in the war, she vividly remembered the home front during World War II. She worked at a local factory that made radio parts for the military. She described the camaraderie among the women workers, how they would sing together during lunch breaks and collect scrap metal for the war effort. She kept a scrapbook from those years with newspaper clippings, ration coupons, and letters from soldiers. She said those years taught her that ordinary people could do extraordinary things when they worked together.",
        },
        {
            "title": "Her Love of Books",
            "date": "",
            "content": "Margaret was an avid reader her entire life. She belonged to three different book clubs and could often be found in her favorite armchair with a book and a cup of tea. Her favorite author was Jane Austen, and she had read Pride and Prejudice over twenty times. She believed reading was the best form of education and always encouraged the grandchildren to read. Her personal library had over five hundred books, and she could tell you the plot and her opinion of every single one.",
        },
    ]

    for mem in sample_memories:
        _ensure_written_memory(db, profile, mem["title"], mem["content"], mem["date"])

    # Document memory: a letter preserved as a .docx.
    _ensure_document_memory(
        db, profile,
        "A Letter from 1948",
        "Dear Margaret,\n\nIt is the summer of 1948, and I cannot believe that in a few short weeks you will be my wife. When I came home from the war two years ago, I never dared to hope I would find someone like you at that church social.\n\nI still remember the way you smiled when we danced to the radio at the diner, and how you laughed when I spilled that strawberry milkshake. You told me every storm runs out of rain, and you have been my sunshine ever since.\n\nI promise to be by your side, to laugh with you, and never to go to bed angry.\n\nWith all my love,\nRobert",
        "1948-06-10",
    )

    # Photograph memory with a vision description.
    _ensure_photo_memory(
        db, profile,
        "Grandma's Garden 1998",
        "An older woman is standing beside a flower bed containing roses near a wooden fence. Behind her is a small garden with tomato plants and tall green grass, and a large tree shades part of the yard. The image has a warm summer light.",
        "1998-07-12",
    )

    # Audio memory with transcript.
    _ensure_audio_memory(
        db, profile,
        "Recorded Interview 2008",
        "Interviewer: Grandma, how did you and Grandpa Robert meet?\n\nMargaret: Well, it was at a church social in 1946, right after the war. He walked in wearing his uniform and I thought to myself, that young man has kind eyes. He asked me to dance and I said yes. We talked for hours that night. On our first date he took me to the diner and bought me a strawberry milkshake. We married in June of 1948 and were together fifty-two years. The secret was never going to bed angry and always finding something to laugh about.",
        "2008-11-03",
    )

    # Video memory (best-effort; skipped gracefully if ffmpeg fails).
    _ensure_video_memory(
        db, profile,
        "Family Reunion 2008",
        "At the family reunion in the summer of 2008, everyone gathered in Grandma's garden. She sat in her favorite chair under the big tree while the grandchildren took turns telling stories. She laughed the loudest of anyone. That evening she made her famous apple pie for the whole family, and the house smelled of cinnamon all night.",
        "2008-08-15",
    )

    _audit(db, demo_user.id, "seed_data", resource_type="system", details="Ensured demo profile and multimodal memories")
    db.close()
    print("Seed data ready.")


if __name__ == "__main__":
    seed()
