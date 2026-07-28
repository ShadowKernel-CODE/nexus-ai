"""
Seed script for MemoryBot.
Creates demo users and a sample memory profile.
Run: python seed.py
"""
import sys
sys.path.insert(0, ".")

from database import init_db, SessionLocal, User, MemoryProfile, MemoryFile, MemoryEmbedding, AuditLog
from auth import hash_password
from rag import generate_embedding
from text_extractor import chunk_text
import json


def seed():
    init_db()
    db = SessionLocal()

    admin = db.query(User).filter(User.email == "admin@memorybot.com").first()
    if not admin:
        admin = User(
            name="Admin",
            email="admin@memorybot.com",
            password_hash=hash_password("admin123"),
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        print(f"Created admin user: admin@memorybot.com / admin123")

    demo_user = db.query(User).filter(User.email == "demo@memorybot.com").first()
    if not demo_user:
        demo_user = User(
            name="Demo User",
            email="demo@memorybot.com",
            password_hash=hash_password("demo123"),
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        print(f"Created demo user: demo@memorybot.com / demo123")

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
            date_of_death="2019-11-22",
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
        print(f"Created memory profile: Margaret Johnson")

        sample_memories = [
            {
                "title": "Childhood during the Depression",
                "content": "Margaret was born in 1925 in a small farming town in Ohio. During the Great Depression, her family lost their farm and had to move to the city. She often told stories about how her mother would stretch a single chicken into meals that lasted three days. Despite the hardships, she always spoke of this time with a sense of resilience and community spirit. Her neighbors would share what little they had, and she learned the value of generosity from watching her parents help others even when they themselves had very little."
            },
            {
                "title": "Meeting Grandpa Robert",
                "content": "Margaret met Robert Johnson at a church social in 1946, just after he returned from serving in World War II. She said he was the most handsome man she had ever seen, with his military uniform and shy smile. Their first date was at the local diner where he bought her a strawberry milkshake. They married in June 1948 and were together for 52 years until Robert passed away in 2000. She often said their secret to a happy marriage was never going to bed angry and always finding something to laugh about."
            },
            {
                "title": "Her Famous Apple Pie",
                "content": "Margaret's apple pie was legendary in the family. The recipe came from her own grandmother and used a secret blend of cinnamon, nutmeg, and a pinch of cardamom. She would pick apples from the tree in her backyard every autumn and spend the whole day baking. The whole house would smell of cinnamon and butter. Every Thanksgiving, she would bake five pies - one for each of her children's families. She never wrote down the recipe, saying it was all in her hands and heart. After she passed, the family tried to recreate it but could never quite get it right."
            },
            {
                "title": "The Garden",
                "content": "Margaret had the most beautiful garden. She grew roses, tulips, and her famous sunflowers that towered over the fence. Every spring she would plan her garden layout like a general planning a campaign. She loved her tomatoes and would can hundreds of jars every summer to last through winter. The neighborhood children would come over to pick fresh vegetables and she would teach them the names of each plant. Her garden was her pride and joy, and she always said it kept her young."
            },
            {
                "title": "Stories of the War Years",
                "content": "Although Margaret herself did not serve in the war, she vividly remembered the home front during World War II. She worked at a local factory that made radio parts for the military. She described the camaraderie among the women workers, how they would sing together during lunch breaks and collect scrap metal for the war effort. She kept a scrapbook from those years with newspaper clippings, ration coupons, and letters from soldiers. She said those years taught her that ordinary people could do extraordinary things when they worked together."
            },
            {
                "title": "Her Love of Books",
                "content": "Margaret was an avid reader her entire life. She belonged to three different book clubs and could often be found in her favorite armchair with a book and a cup of tea. Her favorite author was Jane Austen, and she had read Pride and Prejudice over twenty times. She believed reading was the best form of education and always encouraged the grandchildren to read. Her personal library had over five hundred books, and she could tell you the plot and her opinion of every single one."
            },
        ]

        for mem in sample_memories:
            chunks = chunk_text(mem["content"])
            file = MemoryFile(
                profile_id=profile.id,
                filename=f"sample_{mem['title'].lower().replace(' ', '_')}.txt",
                original_name=f"{mem['title']}.txt",
                file_type=".txt",
                file_size=len(mem["content"]),
                extracted_text=mem["content"],
                text_chunks=json.dumps(chunks),
            )
            db.add(file)
            db.commit()
            db.refresh(file)

            for i, chunk in enumerate(chunks):
                emb_vec = generate_embedding(chunk)
                embedding = MemoryEmbedding(
                    profile_id=profile.id,
                    file_id=file.id,
                    content=chunk,
                    embedding=json.dumps(emb_vec) if emb_vec else "[]",
                    chunk_index=i,
                )
                db.add(embedding)

            db.commit()
            print(f"  Added memory: {mem['title']}")

        log = AuditLog(
            user_id=demo_user.id,
            action="seed_data",
            resource_type="system",
            details="Seeded demo profile and memories",
        )
        db.add(log)
        db.commit()
        print("Seed data created successfully!")
    else:
        print("Demo profile already exists, skipping seed.")

    db.close()


if __name__ == "__main__":
    seed()
