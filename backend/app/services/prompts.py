SYSTEM_PROMPT = (
    "You are a careful assistant that answers only from the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Always cite sources using [n] format tied to the context blocks."
)


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        document_name = chunk.get("document_name") or "document"
        parts.append(
            f"[{idx}] (doc: {document_name}, page: {chunk.get('page_number')})\n"
            f"{chunk.get('text')}"
        )
    return "\n\n".join(parts)
