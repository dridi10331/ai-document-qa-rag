from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "RAG Document Q&A"
    environment: str = "dev"
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    db_url: str = "sqlite:///./data/app.db"
    max_upload_mb: int = 50
    cors_origins_str: str = "http://localhost:3000"
    
    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_str.split(",") if item.strip()]
    enable_ocr: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    llm_backend: str = "groq"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_backend: str = "hf"
    embeddings_device: str = "cpu"
    embeddings_batch_size: int = 32
    vector_index_path: Path = Path("data/processed/faiss.index")
    vector_meta_path: Path = Path("data/processed/faiss_meta.json")
    bm25_path: Path = Path("data/processed/bm25.pkl")
    enable_hybrid_search: bool = True
    enable_query_expansion: bool = True
    query_expansion_max: int = 3
    bm25_weight: float = 0.35
    vector_weight: float = 0.65
    max_context_chunks: int = 6
    max_context_tokens: int = 1800
    chunk_min_words: int = 120
    chunk_max_words: int = 320
    chunk_window_words: int = 260
    chunk_overlap_words: int = 60
    enable_analytics: bool = True
    cost_input_per_1k: float = 0.005
    cost_output_per_1k: float = 0.015

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def empty_api_key_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

# Day 4: cost tracking config
