from __future__ import annotations

from collections import Counter

from sqlalchemy import func
from sqlmodel import Session, select

from app.db.models import Document, DocumentUsage, QueryLog
from app.schemas.analytics import AnalyticsSummary, DocumentStat, QuestionStat


def record_query_summary(session: Session) -> AnalyticsSummary:
    query_count = session.exec(select(func.count(QueryLog.id))).one() or 0
    avg_latency = session.exec(select(func.avg(QueryLog.latency_ms))).one()
    total_cost = session.exec(select(func.sum(QueryLog.cost_estimate))).one() or 0.0

    question_rows = session.exec(select(QueryLog.query)).all()
    question_counts = Counter(question_rows)
    top_questions = [
        QuestionStat(query=query, count=count)
        for query, count in question_counts.most_common(5)
    ]

    usage_rows = session.exec(
        select(DocumentUsage.document_id, func.sum(DocumentUsage.chunk_count))
        .group_by(DocumentUsage.document_id)
        .order_by(func.sum(DocumentUsage.chunk_count).desc())
        .limit(5)
    ).all()

    doc_ids = [row[0] for row in usage_rows]
    doc_lookup = {}
    if doc_ids:
        docs = session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
        doc_lookup = {doc.id: doc.filename for doc in docs}

    top_documents = [
        DocumentStat(
            document_id=row[0],
            filename=doc_lookup.get(row[0], "unknown"),
            usage_count=int(row[1] or 0),
        )
        for row in usage_rows
    ]

    return AnalyticsSummary(
        query_count=int(query_count),
        avg_latency_ms=float(avg_latency or 0.0),
        top_questions=top_questions,
        top_documents=top_documents,
        total_cost=float(total_cost or 0.0),
    )
