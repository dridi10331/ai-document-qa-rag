from pydantic import BaseModel


class StatusEvent(BaseModel):
    document_id: str
    status: str
    detail: str | None = None
