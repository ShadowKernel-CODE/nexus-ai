"""
Seed script for MemoryBot.
Creates demo users and a polished multimodal memory profile for Spider-Man.
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


def _make_demo_spiderman_image(path: str):
    """Draw a New York rooftop with a web and spider so the demo photo resembles Spider-Man."""
    import math
    from PIL import Image, ImageDraw
    width, height = 640, 480
    img = Image.new("RGB", (width, height))
    d = ImageDraw.Draw(img)
    # red-to-blue gradient like the suit.
    for y in range(height):
        t = y / height
        r = int(170 + (20 - 170) * t)
        g = int(20 + (30 - 20) * t)
        b = int(30 + (160 - 30) * t)
        d.line([(0, y), (width, y)], fill=(r, g, b))
    # city skyline silhouette.
    for x, w, h in [(20, 60, 130), (90, 70, 90), (170, 60, 170), (240, 80, 100),
                    (330, 70, 150), (410, 90, 80), (500, 70, 120), (570, 70, 60)]:
        d.rectangle([x, height - h, x + w, height], fill=(12, 12, 26))
    # web.
    cx, cy = width // 2, height // 2 - 50
    radius = 280
    n_lines = 12
    for i in range(n_lines):
        ang = 2 * math.pi * i / n_lines
        d.line([cx, cy, cx + radius * math.cos(ang), cy + radius * math.sin(ang)], fill=(235, 235, 245), width=2)
    for r in range(40, radius + 1, 40):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(235, 235, 245), width=2)
    # spider.
    sx, sy = cx - 70, cy + 100
    d.ellipse([sx - 9, sy - 7, sx + 9, sy + 7], fill=(6, 6, 12))
    d.ellipse([sx - 5, sy - 12, sx + 5, sy - 2], fill=(6, 6, 12))
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1.5), (0, 1.5), (-1.5, 0), (1.5, 0)]:
        d.line([sx, sy, sx + dx * 26, sy + dy * 26], fill=(6, 6, 12), width=2)
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


def _ensure_photo_memory(db, profile, title, description, memory_date="", caption="", draw_fn=None):
    original = f"{title}.jpg"
    existing = db.query(MemoryFile).filter(
        MemoryFile.profile_id == profile.id, MemoryFile.original_name == original
    ).first()
    if existing:
        return existing

    safe = title.lower().replace(" ", "_").replace("'", "")
    filename = f"demo_{safe}.jpg"
    path = os.path.join(settings.UPLOAD_DIR, filename)
    (draw_fn or _make_demo_garden_image)(path)

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
        caption=caption,
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


def _ensure_audio_memory(db, profile, title, transcript, memory_date="", caption=""):
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
        caption=caption,
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


def _ensure_video_memory(db, profile, title, transcript, memory_date="", caption=""):
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
        caption=caption,
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

    # Replace the old demo companion with Spider-Man.
    old_profile = db.query(MemoryProfile).filter(
        MemoryProfile.user_id == demo_user.id,
        MemoryProfile.name == "Margaret Johnson",
    ).first()
    if old_profile:
        db.delete(old_profile)
        db.commit()
        print("Removed old demo companion: Margaret Johnson")

    profile = db.query(MemoryProfile).filter(
        MemoryProfile.user_id == demo_user.id,
        MemoryProfile.name == "Spider-Man",
    ).first()
    if not profile:
        profile = MemoryProfile(
            user_id=demo_user.id,
            name="Spider-Man",
            description="Peter Parker — the friendly neighborhood Spider-Man from Queens, New York. A brilliant science student who was bitten by a spider, gained amazing powers, and chose to use them to protect the little guy. Quick with a joke, even quicker with a web line.",
            relationship_type="Friend",
            date_of_birth="2001-08-10",
            voice_id="",
            personality_traits=["Witty", "Responsible", "Brilliant", "Compassionate", "Self-deprecating", "Hopeful", "Determined"],
            favorite_phrases=["With great power comes great responsibility", "Just your friendly neighborhood Spider-Man", "I'm always gonna be Spider-Man", "Anyone can wear the mask", "A kid from Queens can be a hero"],
            interests=["Science and engineering", "Photography", "Web-swinging", "New York City", "Gadgets and computers", "Helping people"],
            speaking_style="Fast, talkative, and full of banter. Cracks jokes even under pressure to hide his nerves, rambles a little when he's excited, and genuinely cares about everyone he helps.",
            writing_style="His photos at the Daily Bugle say more than his words, but his captions are energetic, a little nerdy, and proudly from Queens.",
            values=["Responsibility", "Protecting the innocent", "Family", "Doing the right thing", "Humility", "Hope"],
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        print("Created memory profile: Spider-Man")

    sample_memories = [
        {
            "title": "The Bite That Changed Everything",
            "date": "2016-05-04",
            "content": "Peter Parker was bitten by a genetically modified spider during a science field trip at the lab. Overnight his body changed — suddenly he had superhuman strength, could stick to walls, and his senses screamed at him a moment before anything dangerous happened. At first he thought it was a crazy allergic reaction. Then he tested his web fluid prototype for real and swung across his first New York rooftop. That's the day the friendly neighborhood Spider-Man was born.",
        },
        {
            "title": "Uncle Ben's Lesson",
            "date": "2016-05-09",
            "content": "The hardest lesson Peter ever learned came with a price. When he let a thief go — someone he could have easily stopped — that same man went on to hurt Uncle Ben, who died saving Peter's life. In that terrible moment Peter understood what Uncle Ben always said: with great power comes great responsibility. From that day on, being Spider-Man was never about fame or money. It was about showing up for people who have no one else.",
        },
        {
            "title": "Life with Aunt May",
            "date": "",
            "content": "After his parents passed, Peter was raised by Uncle Ben and Aunt May in Forest Hills, Queens. Aunt May still makes him dinner, worries about him nonstop, and has no idea her nephew is Spider-Man — or at least she pretends not to. Peter tries so hard to hide the bruises and the late nights, and she acts like she buys every excuse, because that's what they do for each other. Home is the one place he can take the mask off.",
        },
        {
            "title": "Friendly Neighborhood Spider-Man",
            "date": "2016-06-01",
            "content": "Before the big leagues, Spider-Man was just the friendly neighborhood hero of Queens and New York City. He swung people to hospitals, stopped muggings, carried groceries up five flights of stairs, and once saved a kid's science fair project from blowing away. He loved the small stuff — the little guy nobody else noticed. Because that's who he is: a kid from Queens who believes anyone can be a hero if they show up and help.",
        },
        {
            "title": "The Daily Bugle Gig",
            "date": "2016-07-15",
            "content": "Peter Parker photographs himself as Spider-Man for the Daily Bugle. It's a whole production — tripod, self-timer, and a bunch of very awkward poses that he retakes until one looks heroic. J. Jonah Jameson still calls Spider-Man a menace and pays him per picture, refusing to believe the wall-crawler isn't a public nuisance. Peter loves it, because the money pays the bills and the job keeps his alter ego alive.",
        },
        {
            "title": "Balancing Two Lives",
            "date": "",
            "content": "Being a hero is less about the suit and more about showing up — even on the days Peter would rather stay in bed. He juggles homework, rent, Aunt May, and saving New York, and honestly some days he barely keeps it together. But every time he sees a scared kid look up at him and calm down, he remembers why he does it. With great power comes great responsibility, and he'd rather be tired and useful than safe and useless.",
        },
    ]

    for mem in sample_memories:
        _ensure_written_memory(db, profile, mem["title"], mem["content"], mem["date"])

    # Document memory: a letter preserved as a .docx.
    _ensure_document_memory(
        db, profile,
        "A Letter to Aunt May",
        "Dear Aunt May,\n\nI know you worry. I see it every time I come home late, or when you look at my hands and ask about the scratches. I'm not great at explaining where I go, and I'm sorry for that — you've always given me everything, and I still can't tell you the whole truth.\n\nBut I want you to know this: everything I do, I do because of you and Ben. You taught me that the people who have nothing are the ones who need someone the most, and that responsibility matters more than what's easy.\n\nOne day I'll tell you everything, I promise. Until then, please don't worry about me. I'm careful — I have to be, because I have you to come home to.\n\nAll my love,\nPeter",
        "2016-08-10",
    )

    # Photograph memory with a vision description.
    _ensure_photo_memory(
        db, profile,
        "Rooftop in Queens 2016",
        "Spider-Man in his red and blue suit stands on a rooftop in Queens, looking out over the New York City skyline at dusk. The city lights glow behind him and a web is strung across the frame. He looks young but steady, one hand raised as if he's about to swing off.",
        "2016-06-01",
        caption="Spider-Man on a rooftop, looking out over the city.",
        draw_fn=_make_demo_spiderman_image,
    )

    # Audio memory with transcript.
    _ensure_audio_memory(
        db, profile,
        "Voice Memo From the Rooftop",
        "Hey, it's me. Just sitting up here on the water tower in Queens, watching the lights. Long night — stopped a purse snatcher, helped a cat out of a tree, talked a kid down from being scared. Aunt May thinks I was at the library. I'm not great at this whole keeping-secrets thing, but I'm getting better.\n\nYou know, sometimes I sit here and I think about Uncle Ben, and I wonder if he'd be proud. I hope so. This whole thing — the webs, the swinging, saving people — it's all because of him. With great power comes great responsibility. I'm trying, Ben. I really am.",
        "2016-06-15",
        caption="Peter's voice memo from the rooftop.",
    )

    # Video memory (best-effort; skipped gracefully if ffmpeg fails).
    _ensure_video_memory(
        db, profile,
        "First Swing Through the City",
        "The first time Peter Parker really let go and swung through the city, it was terrifying and amazing. He launched off a rooftop, felt his stomach drop, and almost smashed into a water tower before his instincts kicked in. Once he got the rhythm — thwip, release, swing, let go — it was the freest he'd ever felt. He landed on a fire escape, laughed until he couldn't breathe, and did it again. That night he knew he was never going to stop.",
        "2016-06-03",
        caption="Peter's first real web-swing across the city.",
    )

    _audit(db, demo_user.id, "seed_data", resource_type="system", details="Ensured demo Spider-Man profile and multimodal memories")
    db.close()
    print("Seed data ready.")


if __name__ == "__main__":
    seed()
