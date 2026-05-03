from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Iterable

import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    dimension: int


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 0

    def _load_model(self) -> None:
        if self.settings.embeddings_backend == "mock":
            self._dimension = 384
            return
        
        # Delay import to avoid TensorFlow loading issues
        import os
        os.environ['TRANSFORMERS_NO_TF'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embeddings model: %s", self.settings.embeddings_model)
        self._model = SentenceTransformer(
            self.settings.embeddings_model,
            device=self.settings.embeddings_device,
        )
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def _mock_embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dim = self.dimension or 384
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype("float32")
            vectors.append(vec.tolist())
        return vectors

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], dimension=self.dimension)
        if self.settings.embeddings_backend == "mock":
            vectors = self._mock_embed(texts)
            return EmbeddingResult(vectors=vectors, dimension=self.dimension)
        if self._model is None:
            self._load_model()
        vectors = self._model.encode(
            texts,
            batch_size=self.settings.embeddings_batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors_list = [vec.tolist() for vec in vectors]
        return EmbeddingResult(vectors=vectors_list, dimension=self.dimension)


_embedding_service: EmbeddingService | None = None


def get_embedding_service(settings: Settings) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service
