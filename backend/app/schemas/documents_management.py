from pydantic import BaseModel


class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: str
