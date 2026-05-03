from pydantic import BaseModel


class QuestionStat(BaseModel):
    query: str
    count: int


class DocumentStat(BaseModel):
    document_id: str
    filename: str
    usage_count: int


class AnalyticsSummary(BaseModel):
    query_count: int
    avg_latency_ms: float | None = None
    top_questions: list[QuestionStat] = []
    top_documents: list[DocumentStat] = []
    total_cost: float = 0.0
