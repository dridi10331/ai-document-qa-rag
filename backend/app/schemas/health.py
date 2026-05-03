from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    vector_index_size: int
    bm25_index_size: int
