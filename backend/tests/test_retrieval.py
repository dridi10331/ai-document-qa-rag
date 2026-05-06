from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings
from app.db.models import Chunk, Document
from app.services import bm25_store, embeddings, vector_store
from app.services.indexing import index_chunks
from app.services.retrieval import retrieve_chunks


def test_retrieval_returns_relevant_chunk(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        vector_index_path=tmp_path / "processed" / "faiss.index",
        vector_meta_path=tmp_path / "processed" / "faiss_meta.json",
        bm25_path=tmp_path / "processed" / "bm25.pkl",
        embeddings_backend="mock",
        embeddings_model="mock",
        llm_backend="mock",
        enable_query_expansion=False,
    )

    embeddings._embedding_service = None
    vector_store._vector_store = None
    bm25_store._bm25_store = None

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        document = Document(
            filename="sample.txt",
            content_type="text/plain",
            status="ready",
        )
        session.add(document)
        session.commit()
        session.refresh(document)

        chunk1 = Chunk(
            document_id=document.id,
            chunk_index=0,
            text="The sky is blue and the sun is bright.",
        )
        chunk2 = Chunk(
            document_id=document.id,
            chunk_index=1,
            text="Bananas are yellow and rich in potassium.",
        )
        session.add(chunk1)
        session.add(chunk2)
        session.commit()

        index_chunks([chunk1, chunk2], settings)

        result = retrieve_chunks(
            session=session,
            query="What color is the sky",
            settings=settings,
            top_k=1,
            document_ids=None,
            use_hybrid=True,
            enable_query_expansion=False,
        )

        assert result.chunks
        assert "sky" in result.chunks[0].text.lower()
