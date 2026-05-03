from pydantic import BaseModel


class RetrievalChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    page_number: int | None = None
    document_name: str | None = None
