from pydantic import BaseModel

from app.schemas.queries import Citation, Usage


class StreamDonePayload(BaseModel):
    answer: str
    citations: list[Citation] = []
    expanded_query: str | None = None
    latency_ms: float | None = None
    model: str | None = None
    usage: Usage | None = None
    session_id: str | None = None
