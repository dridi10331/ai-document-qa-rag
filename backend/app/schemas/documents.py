from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    size_bytes: int
    page_count: int
    chunk_count: int
    created_at: datetime


class DocumentDetailOut(DocumentOut):
    content_type: str | None = None
    updated_at: datetime
    status_detail: str | None = None


class DocumentChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None = None
    metadata: dict


class DocumentIngestResponse(BaseModel):
    document: DocumentOut
    warnings: list[str] = []
