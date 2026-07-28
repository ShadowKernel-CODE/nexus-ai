import json
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db, User, MemoryProfile, Conversation, Message
from auth import get_user_from_request
from rag import stream_rag_response, search_similar_memories, build_profile_context

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    conversations = []
    if profiles:
        profile_ids = [p.id for p in profiles]
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user.id,
            Conversation.profile_id.in_(profile_ids),
        ).order_by(Conversation.updated_at.desc()).limit(20).all()
    return templates.TemplateResponse("chat.html", {
        "request": request, "user": user, "profiles": profiles,
        "conversations": conversations, "active_profile_id": None,
        "active_conversation": None,
    })


@router.get("/{profile_id}", response_class=HTMLResponse)
async def chat_profile(request: Request, profile_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return RedirectResponse(url="/chat", status_code=302)
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id,
        Conversation.profile_id == profile_id,
    ).order_by(Conversation.updated_at.desc()).limit(20).all()
    return templates.TemplateResponse("chat.html", {
        "request": request, "user": user, "profiles": profiles,
        "conversations": conversations, "active_profile_id": profile_id,
        "active_conversation": None,
    })


@router.get("/{profile_id}/conversation/{conversation_id}", response_class=HTMLResponse)
async def chat_conversation(request: Request, profile_id: str, conversation_id: str, db: Session = Depends(get_db)):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not conversation:
        return RedirectResponse(url="/chat", status_code=302)
    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    profiles = db.query(MemoryProfile).filter(MemoryProfile.user_id == user.id).all()
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id,
        Conversation.profile_id == profile_id,
    ).order_by(Conversation.updated_at.desc()).limit(20).all()
    messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    return templates.TemplateResponse("chat.html", {
        "request": request, "user": user, "profiles": profiles,
        "conversations": conversations, "active_profile_id": profile_id,
        "active_conversation": conversation, "messages": messages,
    })


@router.post("/message")
async def send_message(request: Request, db: Session = Depends(get_db)):
    from database import SessionLocal

    body = await request.json()
    profile_id = body.get("profile_id")
    conversation_id = body.get("conversation_id")
    content = body.get("content", "").strip()

    def sse_error(msg):
        def gen():
            yield f"data: {json.dumps({'error': msg, 'done': True, 'conversation_id': conversation_id})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    if not content:
        return sse_error("Empty message")

    user = get_user_from_request(request, db)
    if not user:
        return sse_error("Unauthorized")

    profile = db.query(MemoryProfile).filter(
        MemoryProfile.id == profile_id, MemoryProfile.user_id == user.id
    ).first()
    if not profile:
        return sse_error("Profile not found")

    if not conversation_id:
        conv = Conversation(user_id=user.id, profile_id=profile_id, title=content[:50])
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id
    else:
        conv = db.query(Conversation).filter(
            Conversation.id == conversation_id, Conversation.user_id == user.id
        ).first()
        if not conv:
            return sse_error("Conversation not found")

    user_msg = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(user_msg)
    db.commit()

    memories = search_similar_memories(db, profile_id, content, limit=10)

    conv_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()
    history = [{"role": m.role, "content": m.content} for m in conv_messages[:-1]]

    saved_profile_id = profile_id
    saved_conversation_id = conversation_id
    saved_memories = memories
    profile_context = build_profile_context(profile)

    def event_stream():
        db = SessionLocal()
        try:
            full_response = ""
            is_error = False
            try:
                for token in stream_rag_response(content, profile_context, saved_memories, history):
                    full_response += token
                    if token.startswith("[ERROR]"):
                        is_error = True
                        yield f"data: {json.dumps({'error': token[8:].strip(), 'done': True, 'conversation_id': saved_conversation_id})}\n\n"
                        return
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                full_response = f"Error generating response: {str(e)}"
                is_error = True
                yield f"data: {json.dumps({'error': full_response, 'done': True, 'conversation_id': saved_conversation_id})}\n\n"
                return

            if not full_response.strip():
                yield f"data: {json.dumps({'error': 'No response generated. Please try again.', 'done': True, 'conversation_id': saved_conversation_id})}\n\n"
                return

            assistant_msg = Message(
                conversation_id=saved_conversation_id,
                role="assistant",
                content=full_response,
                sources=json.dumps([m["content"][:200] for m in saved_memories[:3]]),
            )
            db.add(assistant_msg)
            db.commit()

            yield f"data: {json.dumps({'done': True, 'conversation_id': saved_conversation_id})}\n\n"
        finally:
            db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{conversation_id}/delete")
async def delete_conversation(request: Request, conversation_id: str, db: Session = Depends(get_db)):
    user = get_user_from_request(request, db)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id, Conversation.user_id == user.id
    ).first()
    if conv:
        profile_id = conv.profile_id
        db.delete(conv)
        db.commit()
        return RedirectResponse(url=f"/chat/{profile_id}", status_code=302)
    return RedirectResponse(url="/chat", status_code=302)
