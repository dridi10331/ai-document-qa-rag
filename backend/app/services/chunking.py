from dataclasses import dataclass
import re

from app.services.parsers import ParsedPage

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    page_number: int | None
    metadata: dict


def sentence_split(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    return sentences if sentences else [text.strip()]


def semantic_chunks(text: str, min_words: int, max_words: int) -> list[str]:
    sentences = sentence_split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if current_words + len(words) > max_words and current_words >= min_words:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = len(words)
        else:
            current.append(sentence)
            current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    if not chunks:
        return [text]
    return chunks


def sliding_window(text: str, window_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, window_words - overlap_words)
    windows: list[str] = []
    for start in range(0, len(words), step):
        end = start + window_words
        window = " ".join(words[start:end])
        if window:
            windows.append(window)
        if end >= len(words):
            break
    return windows


def chunk_document(
    pages: list[ParsedPage],
    min_words: int = 120,
    max_words: int = 320,
    window_words: int = 260,
    overlap_words: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        base_chunks = semantic_chunks(page.text, min_words=min_words, max_words=max_words)
        for base_text in base_chunks:
            for window_text in sliding_window(
                base_text,
                window_words=window_words,
                overlap_words=overlap_words,
            ):
                chunks.append(
                    Chunk(
                        text=window_text,
                        page_number=page.page_number,
                        metadata=page.metadata.copy(),
                    )
                )
    return chunks
