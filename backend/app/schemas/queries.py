from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: list[str] | None = None
    session_id: str | None = None
    use_hybrid: bool | None = None
    enable_query_expansion: bool | None = None


class Citation(BaseModel):
    document_id: str
    document_name: str | None = None
    chunk_id: str | None = None
    chunk_index: int
    page_number: int | None = None
    score: float | None = None
    text: str | None = None


class Usage(BaseModel):
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_estimate: float | None = None


class QueryResponse(BaseModel):
    session_id: str | None = None
    expanded_query: str | None = None
    answer: str
    citations: list[Citation] = []
    latency_ms: float | None = None
    model: str | None = None
    usage: Usage | None = None
