# MemoryBot - AI Memory Companion

A Python/FastAPI rebuild of MemoryBot, an AI-powered application that helps people preserve and interact with memories of loved ones.

## Features

- **User Authentication** - Register/login with secure password hashing and JWT sessions
- **Memory Profiles** - Create profiles for loved ones with biographical details
- **File Upload & Text Extraction** - Upload PDFs, DOCX, TXT, images, audio, and video files. Text is extracted and chunked for RAG.
- **AI-Powered Chat** - Streaming chat with RAG (Retrieval-Augmented Generation) using OpenAI/OpenRouter
- **Vector Search** - Semantic search through memories using embeddings or keyword fallback
- **Voice Features** - Text-to-speech via edge-tts, Speech-to-text via OpenAI Whisper
- **Memory Timeline** - Browse uploaded files and memories
- **Admin Dashboard** - System stats, user management, data export, audit logging

## Tech Stack

- **Backend**: FastAPI + SQLAlchemy + SQLite (PostgreSQL for production)
- **Auth**: bcrypt + JWT (python-jose)
- **AI**: OpenAI/OpenRouter for LLM chat and embeddings
- **Voice**: edge-tts (TTS), OpenAI Whisper API (STT)
- **Frontend**: Jinja2 templates + Tailwind CSS (CDN)
- **File Processing**: PyPDF2, python-docx, custom text extractor

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your API keys

# 4. Seed database with demo data
python seed.py

# 5. Run the server
python main.py
# Or: uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Demo Accounts

| Email | Password | Notes |
|-------|----------|-------|
| admin@memorybot.com | admin123 | Admin user |
| demo@memorybot.com | demo123 | Has sample "Margaret Johnson" profile with 6 memories |

## AI Configuration

Set your OpenAI or OpenRouter API key in `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
CHAT_MODEL=openai/gpt-4o-mini
```

Without an API key, the app works in **fallback mode** with rule-based responses.

## Project Structure

```
memorybot/
├── main.py              # FastAPI application entry point
├── config.py            # Settings (pydantic-settings)
├── database.py          # SQLAlchemy models
├── auth.py              # Authentication utilities
├── rag.py               # RAG pipeline, embeddings, search
├── text_extractor.py    # File text extraction & chunking
├── seed.py              # Database seeder
├── routes_auth.py       # /auth/* routes
├── routes_profiles.py   # /profiles/* routes (CRUD, upload)
├── routes_chat.py       # /chat/* routes (streaming SSE)
├── routes_search.py     # /search routes
├── routes_admin.py      # /admin/* routes
├── routes_voice.py      # /voice/* routes (TTS, STT)
├── templates/           # Jinja2 HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── profiles.html
│   ├── profile_detail.html
│   ├── chat.html
│   ├── search.html
│   ├── admin.html
│   └── partials/
│       └── sidebar.html
├── static/              # Static assets
├── uploads/             # User uploaded files
└── .env                 # Environment config
```

## Important Disclaimer

**MemoryBot is an AI simulation.** The AI is NOT the actual person being remembered. Responses are generated based on uploaded memories and context. This tool is meant to help people preserve and explore memories, not to replicate a person.
