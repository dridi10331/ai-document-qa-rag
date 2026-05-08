from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Iterable

import numpy as np
import requests

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
        return self._dimension or 384

    def _load_model(self) -> None:
        backend = self.settings.embeddings_backend

        if backend == "mock":
            self._dimension = 384
            return

        if backend == "groq":
            # Groq embeddings - no local model needed
            self._dimension = 1024  # nomic-embed-text-v1.5 dimension
            return

        if backend == "hf":
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
            return

        # Default fallback
        self._dimension = 384

    def _mock_embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dim = self.dimension or 384
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype("float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors

    def _groq_embed(self, texts: list[str]) -> list[list[float]]:
        """Use Groq's embedding API (nomic-embed-text-v1.5)."""
        if not self.settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set, falling back to mock embeddings")
            return self._mock_embed(texts)

        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        vectors = []
        # Groq embeddings API processes one at a time
        for text in texts:
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/embeddings",
                    headers=headers,
                    json={
                        "model": "nomic-embed-text-v1.5",
                        "input": text[:8192],  # max tokens
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                vectors.append(data["data"][0]["embedding"])
            except Exception as e:
                logger.warning("Groq embedding failed for text, using mock: %s", e)
                vectors.extend(self._mock_embed([text]))
        return vectors

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], dimension=self.dimension)

        backend = self.settings.embeddings_backend

        if backend == "groq":
            if self._dimension is None:
                self._load_model()
            vectors = self._groq_embed(texts)
            return EmbeddingResult(vectors=vectors, dimension=self.dimension)

        if backend == "mock":
            vectors = self._mock_embed(texts)
            return EmbeddingResult(vectors=vectors, dimension=self.dimension)

        # HuggingFace local
        if self._model is None:
            self._load_model()
        if self._model is None:
            # Fallback to mock if model failed to load
            vectors = self._mock_embed(texts)
            return EmbeddingResult(vectors=vectors, dimension=self.dimension)

        raw = self._model.encode(
            texts,
            batch_size=self.settings.embeddings_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors_list = [vec.tolist() for vec in raw]
        return EmbeddingResult(vectors=vectors_list, dimension=self.dimension)


_embedding_service: EmbeddingService | None = None


def get_embedding_service(settings: Settings) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service
