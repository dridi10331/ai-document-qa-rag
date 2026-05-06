from app.services.chunking import chunk_document
from app.services.parsers import ParsedPage


def test_chunk_document_creates_windows() -> None:
    page = ParsedPage(
        text="Sentence one. Sentence two. Sentence three.",
        page_number=1,
        metadata={"page": 1},
    )
    chunks = chunk_document(
        [page],
        min_words=1,
        max_words=6,
        window_words=4,
        overlap_words=1,
    )
    assert chunks
    assert all(chunk.page_number == 1 for chunk in chunks)
