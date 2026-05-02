from uuid import uuid4

from fastapi import UploadFile
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.db.crud import add_chunks, create_document, update_document
from app.db.models import Chunk, Document
from app.schemas.status import StatusEvent
from app.services.chunking import chunk_document
from app.services.indexing import index_chunks
from app.services.status import status_hub
from app.services.parsers import parse_document
from app.utils.file_utils import sanitize_filename, save_upload_file


async def ingest_upload(
    upload_file: UploadFile,
    session: Session,
    settings: Settings,
) -> tuple[Document, list[str]]:
    if not upload_file.filename:
        raise ValueError("File has no name.")

    safe_name = sanitize_filename(upload_file.filename)
    document = Document(
        id=str(uuid4()),
        filename=safe_name,
        content_type=upload_file.content_type,
        status="processing",
        status_detail="uploading",
    )
    create_document(session, document)
    await status_hub.publish(
        StatusEvent(document_id=document.id, status="processing", detail="uploading")
    )

    raw_path = settings.raw_dir / f"{document.id}_{safe_name}"
    size_bytes = await save_upload_file(upload_file, raw_path)
    update_document(session, document, size_bytes=size_bytes)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        update_document(
            session,
            document,
            status="failed",
            status_detail="file too large",
        )
        raise ValueError("File exceeds max_upload_mb.")

    update_document(session, document, status_detail="parsing")
    await status_hub.publish(
        StatusEvent(document_id=document.id, status="processing", detail="parsing")
    )
    parsed = await run_in_threadpool(parse_document, raw_path, settings.enable_ocr)

    update_document(session, document, status_detail="chunking")
    await status_hub.publish(
        StatusEvent(document_id=document.id, status="processing", detail="chunking")
    )
    chunks = await run_in_threadpool(
        chunk_document,
        parsed.pages,
        settings.chunk_min_words,
        settings.chunk_max_words,
        settings.chunk_window_words,
        settings.chunk_overlap_words,
    )

    db_chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=index,
            text=chunk.text,
            page_number=chunk.page_number,
            chunk_metadata=chunk.metadata,
        )
        for index, chunk in enumerate(chunks)
    ]
    add_chunks(session, db_chunks)

    update_document(session, document, status_detail="indexing")
    await status_hub.publish(
        StatusEvent(document_id=document.id, status="processing", detail="indexing")
    )
    await run_in_threadpool(
        index_chunks,
        db_chunks,
        settings,
    )

    update_document(
        session,
        document,
        status="ready",
        status_detail="complete",
        page_count=len(parsed.pages),
        chunk_count=len(db_chunks),
    )
    await status_hub.publish(
        StatusEvent(document_id=document.id, status="ready", detail="complete")
    )

    warnings: list[str] = []
    if not settings.enable_ocr:
        empty_pages = [page for page in parsed.pages if not page.text.strip()]
        if empty_pages:
            warnings.append("OCR disabled: some pages may be empty.")

    return document, warnings
