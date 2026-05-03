from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from app.core.config import Settings


@dataclass
class VectorSearchResult:
    chunk_id: str
    score: float


class FaissStore:
    def __init__(self, index_path: Path, meta_path: Path) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []
        self.dimension: int | None = None
        self._lock = Lock()

    def load_or_create(self, dimension: int) -> None:
        with self._lock:
            if self.index is not None:
                return
            if self.index_path.exists() and self.meta_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.chunk_ids = meta.get("chunk_ids", [])
                self.dimension = int(meta.get("dim", dimension))
                return
            self.index = faiss.IndexFlatIP(dimension)
            self.chunk_ids = []
            self.dimension = dimension
            self._persist()

    def _persist(self) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        payload = {"dim": self.dimension, "chunk_ids": self.chunk_ids}
        self.meta_path.write_text(json.dumps(payload), encoding="utf-8")

    def size(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def add(self, vectors: list[list[float]], chunk_ids: list[str]) -> None:
        if not vectors:
            return
        if len(vectors) != len(chunk_ids):
            raise ValueError("Vector count and chunk id count must match.")
        with self._lock:
            if self.index is None:
                raise ValueError("FAISS index not initialized.")
            if self.dimension is None:
                self.dimension = len(vectors[0])
            matrix = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
            self.chunk_ids.extend(chunk_ids)
            self._persist()

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        if self.index is None or not self.chunk_ids:
            return []
        query = np.asarray([vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        results: list[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append(VectorSearchResult(chunk_id=self.chunk_ids[idx], score=float(score)))
        return results


_vector_store: FaissStore | None = None


def get_vector_store(settings: Settings, dimension: int) -> FaissStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FaissStore(settings.vector_index_path, settings.vector_meta_path)
    _vector_store.load_or_create(dimension)
    return _vector_store
