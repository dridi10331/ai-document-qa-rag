from __future__ import annotations

import json
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.db.crud import (
    add_document_usage,
    create_query_log,
    delete_chunks_by_document,
    delete_document,
    get_document,
    list_chat_sessions,
    list_chunks_by_document,
    list_documents,
)
from app.db.models import DocumentUsage, QueryLog
from app.db.session import get_session
from app.schemas.analytics import AnalyticsSummary
from app.schemas.chat import ChatMessageOut, ChatSessionCreate, ChatSessionOut
from app.schemas.documents import DocumentChunkOut, DocumentDetailOut, DocumentIngestResponse, DocumentOut
from app.schemas.documents_management import DocumentDeleteResponse
from app.schemas.documents_status import DocumentStatusOut
from app.schemas.health import HealthStatus
from app.schemas.queries import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.queries_stream import StreamDonePayload
from app.services.analytics import record_query_summary
from app.services.chat import add_message, create_session, get_history
from app.services.health import get_index_sizes
from app.services.indexing import rebuild_indexes
from app.services.ingest import ingest_upload
from app.services.llm import estimate_cost, estimate_prompt_tokens, generate_answer, stream_answer
from app.services.retrieval import retrieve_chunks
from app.services.status import status_hub
from app.utils.text_utils import approx_token_count

router = APIRouter()


def _build_citations(chunks) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        citations.append(
            Citation(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                score=chunk.score,
                text=chunk.text[:240],
            )
        )
    return citations


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    settings = get_settings()
    vector_size, bm25_size = get_index_sizes(settings)
    return HealthStatus(status="ok", vector_index_size=vector_size, bm25_index_size=bm25_size)


@router.post("/documents/upload", response_model=list[DocumentIngestResponse])
async def upload_documents(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> list[DocumentIngestResponse]:
    settings = get_settings()
    responses: list[DocumentIngestResponse] = []

    for upload in files:
        try:
            document, warnings = await ingest_upload(upload, session, settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        responses.append(
            DocumentIngestResponse(
                document=DocumentOut.model_validate(document), warnings=warnings
            )
        )
    return responses


@router.get("/documents", response_model=list[DocumentOut])
def list_all_documents(session: Session = Depends(get_session)) -> list[DocumentOut]:
    documents = list_documents(session)
    return [DocumentOut.model_validate(doc) for doc in documents]


@router.get("/documents/{document_id}", response_model=DocumentDetailOut)
def get_document_by_id(
    document_id: str,
    session: Session = Depends(get_session),
) -> DocumentDetailOut:
    document = get_document(session, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetailOut.model_validate(document)


@router.get("/documents/{document_id}/chunks", response_model=list[DocumentChunkOut])
def list_document_chunks(
    document_id: str,
    session: Session = Depends(get_session),
) -> list[DocumentChunkOut]:
    document = get_document(session, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = list_chunks_by_document(session, document_id)
    return [DocumentChunkOut.model_validate(chunk) for chunk in chunks]


@router.get("/documents/{document_id}/status", response_model=DocumentStatusOut)
def get_document_status(
    document_id: str,
    session: Session = Depends(get_session),
) -> DocumentStatusOut:
    document = get_document(session, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusOut(
        document_id=document.id,
        status=document.status,
        status_detail=document.status_detail,
    )


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document_by_id(
    document_id: str,
    session: Session = Depends(get_session),
) -> DocumentDeleteResponse:
    document = get_document(session, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_chunks_by_document(session, document_id)
    delete_document(session, document)
    settings = get_settings()
    await run_in_threadpool(rebuild_indexes, session, settings)
    return DocumentDeleteResponse(document_id=document_id, status="deleted")


@router.post("/query", response_model=QueryResponse)
def query_documents(
    payload: QueryRequest,
    session: Session = Depends(get_session),
) -> QueryResponse:
    settings = get_settings()
    start = perf_counter()

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = payload.session_id
    if session_id is None:
        session_id = create_session(session, payload.query[:60]).id

    retrieval = retrieve_chunks(
        session=session,
        query=payload.query,
        settings=settings,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        use_hybrid=payload.use_hybrid,
        enable_query_expansion=payload.enable_query_expansion,
    )

    history = get_history(session, session_id) if session_id else []
    history_payload = [
        {"role": msg.role, "content": msg.content} for msg in history[-6:]
    ]

    context_chunks = [chunk.model_dump() for chunk in retrieval.chunks]
    answer_result = generate_answer(payload.query, context_chunks, history_payload, settings)
    latency_ms = (perf_counter() - start) * 1000
    citations = _build_citations(retrieval.chunks)

    tokens_in = answer_result.usage.tokens_in if answer_result.usage else None
    tokens_out = answer_result.usage.tokens_out if answer_result.usage else None
    if tokens_in is None:
        tokens_in = estimate_prompt_tokens(payload.query, context_chunks, settings)
    if tokens_out is None:
        tokens_out = approx_token_count(answer_result.answer)
    cost = estimate_cost(settings, tokens_in, tokens_out)

    if session_id:
        add_message(session, session_id, "user", payload.query)
        add_message(session, session_id, "assistant", answer_result.answer)

    if settings.enable_analytics:
        log = QueryLog(
            session_id=session_id,
            query=payload.query,
            expanded_query=retrieval.expanded_query,
            top_k=payload.top_k,
            document_ids=",".join(payload.document_ids or []),
            latency_ms=latency_ms,
            model=answer_result.model,
            answer_chars=len(answer_result.answer),
            citations_count=len(citations),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_estimate=cost,
        )
        log = create_query_log(session, log)
        usages = [
            DocumentUsage(query_id=log.id, document_id=chunk.document_id, chunk_count=1)
            for chunk in retrieval.chunks
        ]
        add_document_usage(session, usages)

    return QueryResponse(
        session_id=session_id,
        expanded_query=retrieval.expanded_query,
        answer=answer_result.answer,
        citations=citations,
        latency_ms=latency_ms,
        model=answer_result.model,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_estimate=cost),
    )


@router.get("/query/stream")
def stream_query(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    document_ids: str | None = None,
    use_hybrid: bool | None = None,
    enable_query_expansion: bool | None = None,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    settings = get_settings()
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    doc_ids = [doc_id for doc_id in document_ids.split(",") if doc_id] if document_ids else None

    if session_id is None:
        session_id = create_session(session, query[:60]).id

    start = perf_counter()
    retrieval = retrieve_chunks(
        session=session,
        query=query,
        settings=settings,
        top_k=top_k,
        document_ids=doc_ids,
        use_hybrid=use_hybrid,
        enable_query_expansion=enable_query_expansion,
    )

    history = get_history(session, session_id) if session_id else []
    history_payload = [
        {"role": msg.role, "content": msg.content} for msg in history[-6:]
    ]
    context_chunks = [chunk.model_dump() for chunk in retrieval.chunks]
    citations = _build_citations(retrieval.chunks)

    def event_generator():
        answer_fragments: list[str] = []
        yield _sse_event("status", {"stage": "retrieval", "chunks": len(context_chunks)})
        for token in stream_answer(query, context_chunks, history_payload, settings):
            answer_fragments.append(token)
            yield _sse_event("token", {"token": token})
        answer = "".join(answer_fragments)
        latency_ms = (perf_counter() - start) * 1000
        tokens_in = estimate_prompt_tokens(query, context_chunks, settings)
        tokens_out = approx_token_count(answer)
        cost = estimate_cost(settings, tokens_in, tokens_out)

        if session_id:
            add_message(session, session_id, "user", query)
            add_message(session, session_id, "assistant", answer)

        # Determine model name based on backend
        if settings.llm_backend == "groq":
            model_name = settings.groq_model
        elif settings.llm_backend == "ollama":
            model_name = settings.ollama_model
        else:
            model_name = "unknown"

        if settings.enable_analytics:
            log = QueryLog(
                session_id=session_id,
                query=query,
                expanded_query=retrieval.expanded_query,
                top_k=top_k,
                document_ids=",".join(doc_ids or []),
                latency_ms=latency_ms,
                model=model_name,
                answer_chars=len(answer),
                citations_count=len(citations),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_estimate=cost,
            )
            log = create_query_log(session, log)
            usages = [
                DocumentUsage(query_id=log.id, document_id=chunk.document_id, chunk_count=1)
                for chunk in retrieval.chunks
            ]
            add_document_usage(session, usages)

        done_payload = StreamDonePayload(
            answer=answer,
            citations=citations,
            expanded_query=retrieval.expanded_query,
            latency_ms=latency_ms,
            model=model_name,
            usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_estimate=cost),
            session_id=session_id,
        )
        yield _sse_event("done", done_payload.model_dump())

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.websocket("/ws/documents/{document_id}")
async def document_status_ws(websocket: WebSocket, document_id: str) -> None:
    await websocket.accept()
    try:
        async for event in status_hub.subscribe(document_id):
            await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        return


@router.post("/sessions", response_model=ChatSessionOut)
def create_chat_session(
    payload: ChatSessionCreate,
    session: Session = Depends(get_session),
) -> ChatSessionOut:
    chat_session = create_session(session, payload.title)
    return ChatSessionOut.model_validate(chat_session)


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(session: Session = Depends(get_session)) -> list[ChatSessionOut]:
    sessions = list_chat_sessions(session)
    return [ChatSessionOut.model_validate(item) for item in sessions]


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_session_messages(
    session_id: str,
    session: Session = Depends(get_session),
) -> list[ChatMessageOut]:
    messages = get_history(session, session_id)
    return [ChatMessageOut.model_validate(message) for message in messages]


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(session: Session = Depends(get_session)) -> AnalyticsSummary:
    return record_query_summary(session)


@router.post("/indexes/rebuild")
async def rebuild_all_indexes(session: Session = Depends(get_session)) -> dict:
    settings = get_settings()
    await run_in_threadpool(rebuild_indexes, session, settings)
    return {"status": "rebuild_started"}


@router.post("/eval/retrieval")
def evaluate_retrieval(
    payload: QueryRequest,
    session: Session = Depends(get_session),
) -> dict:
    """
    Retrieval evaluation endpoint.

    Returns LLM-judged relevance metrics:
    - Precision@k: fraction of top-k chunks judged relevant
    - MRR: rank of first relevant result
    - Rerank gain: precision improvement from reranking
    - Score distribution: raw retrieval scores

    Note: Uses Groq LLM as relevance judge (no labeled ground truth required).
    For production evaluation, use RAGAS or DeepEval with labeled datasets.
    """
    import time
    from app.services.evaluation import evaluate_retrieval as eval_fn
    from app.services.retrieval import retrieve_chunks

    settings = get_settings()

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.perf_counter()

    # Retrieve without reranking first (for gain comparison)
    retrieval_base = retrieve_chunks(
        session=session,
        query=payload.query,
        settings=settings,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        use_hybrid=payload.use_hybrid,
        enable_query_expansion=False,
    )

    # Retrieve with full pipeline
    retrieval_full = retrieve_chunks(
        session=session,
        query=payload.query,
        settings=settings,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        use_hybrid=payload.use_hybrid,
        enable_query_expansion=payload.enable_query_expansion,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    metrics = eval_fn(
        query=payload.query,
        chunks_before_rerank=retrieval_base.chunks,
        chunks_after_rerank=retrieval_full.chunks,
        expanded_query=retrieval_full.expanded_query,
        reranked=retrieval_full.reranked,
        latency_ms=latency_ms,
        settings=settings,
    )

    return {
        "query": metrics.query,
        "expanded_query": metrics.expanded_query,
        "pipeline": {
            "query_expansion": metrics.expanded_query is not None,
            "hybrid_search": payload.use_hybrid if payload.use_hybrid is not None else settings.enable_hybrid_search,
            "reranking_applied": metrics.reranked,
        },
        "retrieval_metrics": {
            "chunks_retrieved": metrics.chunks_retrieved,
            "avg_score": metrics.avg_score,
            "max_score": metrics.max_score,
            "min_score": metrics.min_score,
            "score_std": metrics.score_std,
        },
        "llm_judged_metrics": {
            "relevance_scores": metrics.relevance_scores,
            "precision_at_k": metrics.precision_at_k,
            "mrr": metrics.mrr,
            "rerank_gain": metrics.rerank_gain,
        },
        "latency_ms": metrics.latency_ms,
        "notes": metrics.notes,
        "chunks": [
            {
                "rank": i + 1,
                "chunk_id": c.chunk_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "retrieval_score": round(c.score, 4) if c.score else None,
                "llm_relevance": metrics.relevance_scores[i] if i < len(metrics.relevance_scores) else None,
                "text_preview": c.text[:200] if c.text else "",
            }
            for i, c in enumerate(retrieval_full.chunks)
        ],
    }
