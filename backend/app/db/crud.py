from datetime import datetime, timezone
from sqlmodel import Session, col, select

from app.db.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    DocumentUsage,
    QueryLog,
)


def create_document(session: Session, document: Document) -> Document:
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def update_document(session: Session, document: Document, **kwargs) -> Document:
    kwargs.setdefault("updated_at", datetime.now(timezone.utc))
    for key, value in kwargs.items():
        setattr(document, key, value)
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def list_documents(session: Session) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list(session.exec(statement))


def list_documents_by_ids(session: Session, document_ids: list[str]) -> list[Document]:
    if not document_ids:
        return []
    statement = select(Document).where(col(Document.id).in_(document_ids))
    return list(session.exec(statement))


def get_document(session: Session, document_id: str) -> Document | None:
    return session.get(Document, document_id)


def delete_document(session: Session, document: Document) -> None:
    session.delete(document)
    session.commit()


def add_chunks(session: Session, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    session.add_all(chunks)
    session.commit()


def delete_chunks_by_document(session: Session, document_id: str) -> None:
    chunks = list_chunks_by_document(session, document_id)
    for chunk in chunks:
        session.delete(chunk)
    session.commit()


def list_chunks_by_document(session: Session, document_id: str) -> list[Chunk]:
    statement = select(Chunk).where(Chunk.document_id == document_id)
    return list(session.exec(statement))


def list_chunks_by_ids(session: Session, chunk_ids: list[str]) -> list[Chunk]:
    if not chunk_ids:
        return []
    statement = select(Chunk).where(col(Chunk.id).in_(chunk_ids))
    return list(session.exec(statement))


def list_all_chunks(session: Session) -> list[Chunk]:
    statement = select(Chunk)
    return list(session.exec(statement))


def create_chat_session(session: Session, chat_session: ChatSession) -> ChatSession:
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


def list_chat_sessions(session: Session) -> list[ChatSession]:
    statement = select(ChatSession).order_by(ChatSession.created_at.desc())
    return list(session.exec(statement))


def add_chat_message(session: Session, message: ChatMessage) -> ChatMessage:
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def list_chat_messages(session: Session, session_id: str) -> list[ChatMessage]:
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return list(session.exec(statement))


def create_query_log(session: Session, log: QueryLog) -> QueryLog:
    session.add(log)
    session.commit()
    session.refresh(log)
    return log


def add_document_usage(session: Session, usages: list[DocumentUsage]) -> None:
    if not usages:
        return
    session.add_all(usages)
    session.commit()
