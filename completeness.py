"""Transparent Memory Profile Completeness scoring.

The score communicates "more preserved information -> richer memory representation".
It is a weighted sum of objective, observable signals. 100% does NOT mean a perfect
recreation of a human being — it means the profile is richly documented.
"""


def compute_completeness(profile, files):
    files = list(files or [])

    # Identity: name, relationship, date of birth, description, photo.
    identity = 0.0
    if profile.name:
        identity += 15
    if profile.relationship_type:
        identity += 25
    if profile.date_of_birth:
        identity += 25
    if profile.description and len(profile.description.strip()) > 20:
        identity += 25
    if profile.photo_url:
        identity += 10

    # Personality: traits, speaking style, favorite phrases.
    personality = 0.0
    traits = profile.personality_traits or []
    personality += min(len(traits), 4) * 15  # up to 60
    if profile.speaking_style:
        personality += 20
    if len(profile.favorite_phrases or []) >= 2:
        personality += 20
    personality = min(personality, 100.0)

    # Life events: dated memories + written stories.
    dated = [f for f in files if f.memory_date]
    written = [f for f in files if f.memory_type == "written"]
    life_events = 0.0
    if files:
        life_events = min(40 + min(len(dated), 6) * 10, 100.0)
        if written:
            life_events = min(life_events + min(len(written), 5) * 4, 100.0)

    # Documents.
    documents = min(len([f for f in files if f.memory_type == "document"]) * 10, 100.0)

    # Photographs.
    photographs = min(len([f for f in files if f.memory_type == "photograph"]) * 20, 100.0)

    # Voice: chosen voice + audio/video memories.
    voice = 0.0
    if profile.voice_id:
        voice += 50
    voice += min(len([f for f in files if f.memory_type in ("audio", "video")]) * 10, 50)

    overall = (
        identity * 0.20
        + personality * 0.20
        + life_events * 0.20
        + documents * 0.15
        + photographs * 0.15
        + voice * 0.10
    )

    return {
        "overall": round(overall),
        "categories": [
            {"key": "identity", "label": "Identity", "value": round(identity)},
            {"key": "personality", "label": "Personality", "value": round(personality)},
            {"key": "life_events", "label": "Life Events", "value": round(life_events)},
            {"key": "documents", "label": "Documents", "value": round(documents)},
            {"key": "photographs", "label": "Photographs", "value": round(photographs)},
            {"key": "voice", "label": "Voice", "value": round(voice)},
        ],
    }
