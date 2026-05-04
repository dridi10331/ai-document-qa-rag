from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from threading import Lock

from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.utils.text_utils import tokenize


@dataclass
class BM25SearchResult:
    chunk_id: str
    score: float


class BM25Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.chunk_ids: list[str] = []
        self.tokenized: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._lock = Lock()

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = pickle.loads(self.path.read_bytes())
        self.chunk_ids = payload.get("chunk_ids", [])
        self.tokenized = payload.get("tokenized", [])
        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunk_ids": self.chunk_ids, "tokenized": self.tokenized}
        self.path.write_bytes(pickle.dumps(payload))

    def size(self) -> int:
        return len(self.chunk_ids)

    def rebuild(self, texts: list[str], chunk_ids: list[str]) -> None:
        with self._lock:
            self.chunk_ids = chunk_ids
            self.tokenized = [tokenize(text) for text in texts]
            self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None
            self._persist()

    def add(self, texts: list[str], chunk_ids: list[str]) -> None:
        if not texts:
            return
        with self._lock:
            self.tokenized.extend([tokenize(text) for text in texts])
            self.chunk_ids.extend(chunk_ids)
            self.bm25 = BM25Okapi(self.tokenized)
            self._persist()

    def search(self, query: str, top_k: int) -> list[BM25SearchResult]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        if scores is None or len(scores) == 0:
            return []
        indexed = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        results: list[BM25SearchResult] = []
        for idx, score in indexed:
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append(BM25SearchResult(chunk_id=self.chunk_ids[idx], score=float(score)))
        return results


_bm25_store: BM25Store | None = None


def get_bm25_store(settings: Settings) -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store(settings.bm25_path)
        _bm25_store.load()
    return _bm25_store
