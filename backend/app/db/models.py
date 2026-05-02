from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    filename: str
    content_type: Optional[str] = None
    size_bytes: int = 0
    status: str = "uploaded"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    page_count: int = 0
    chunk_count: int = 0
    status_detail: Optional[str] = None


class Chunk(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QueryLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: Optional[str] = Field(default=None, index=True)
    query: str
    expanded_query: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    top_k: int = 5
    document_ids: Optional[str] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    answer_chars: int = 0
    citations_count: int = 0
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_estimate: Optional[float] = None


class DocumentUsage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    query_id: str = Field(foreign_key="querylog.id", index=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    chunk_count: int = 0
