from sqlmodel import Session

from app.db.crud import add_chat_message, create_chat_session, list_chat_messages
from app.db.models import ChatMessage, ChatSession


def create_session(session: Session, title: str | None) -> ChatSession:
    chat_session = ChatSession(title=title)
    return create_chat_session(session, chat_session)


def add_message(session: Session, session_id: str, role: str, content: str) -> ChatMessage:
    message = ChatMessage(session_id=session_id, role=role, content=content)
    return add_chat_message(session, message)


def get_history(session: Session, session_id: str) -> list[ChatMessage]:
    return list_chat_messages(session, session_id)
