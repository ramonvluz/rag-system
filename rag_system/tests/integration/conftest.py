# rag_system/tests/integration/conftest.py

"""Fixtures compartilhadas pelos testes de integração.

Estratégia de mock:
- Componentes leves (TextCleaner, ParagraphChunker, BM25Search,
  HybridSearch, PromptBuilder) rodam reais — validam lógica de negócio.
- Componentes pesados (SentenceTransformer, CrossEncoder) e APIs
  externas (Groq, Ollama) são mockados — evitam dependências de rede
  e GPU em CI.
- ChromaDB usa diretório temporário via tmp_path para isolamento total
  entre testes, sem poluir o ChromaDB de produção.
"""
import pytest
from unittest.mock import MagicMock
from rag_system.core.models import Document, Chunk


@pytest.fixture
def sample_document(tmp_path):
    """Document sintético representando um CV simples."""
    return Document(
        doc_id="ramon_cv_test1234",
        source_uri=str(tmp_path / "ramon_cv.pdf"),
        text=(
            "Ramon Valgas Luz é Engenheiro de IA com experiência em sistemas RAG.\n\n"
            "Trabalhou como Coordenador de Marketing na Studio 235.\n\n"
            "Habilidades: Python, LangChain, ChromaDB, FastAPI, RAGAS."
        ),
        metadata={"filename": "ramon_cv.pdf", "filetype": "pdf"},
    )


@pytest.fixture
def sample_chunks():
    """Chunks pré-processados para testes que pulam o chunking."""
    return [
        Chunk(
            chunk_id="ramon_cv_test1234_chunk_0000",
            doc_id="ramon_cv_test1234",
            text="Ramon Valgas Luz é Engenheiro de IA com experiência em sistemas RAG.",
            metadata={"filename": "ramon_cv.pdf", "source_uri": "ramon_cv.pdf"},
            embedding=[0.1] * 768,
        ),
        Chunk(
            chunk_id="ramon_cv_test1234_chunk_0001",
            doc_id="ramon_cv_test1234",
            text="Trabalhou como Coordenador de Marketing na Studio 235.",
            metadata={"filename": "ramon_cv.pdf", "source_uri": "ramon_cv.pdf"},
            embedding=[0.2] * 768,
        ),
        Chunk(
            chunk_id="ramon_cv_test1234_chunk_0002",
            doc_id="ramon_cv_test1234",
            text="Habilidades: Python, LangChain, ChromaDB, FastAPI, RAGAS.",
            metadata={"filename": "ramon_cv.pdf", "source_uri": "ramon_cv.pdf"},
            embedding=[0.3] * 768,
        ),
    ]


@pytest.fixture
def mock_embedder():
    """BGEEmbedder mockado: embed_chunks preenche embeddings fixos, embed_query retorna vetor fixo."""
    embedder = MagicMock()
    # setattr retorna None — o `or c` garante que o chunk seja incluído na lista resultante
    embedder.embed_chunks.side_effect = lambda chunks: [
        setattr(c, "embedding", [0.1] * 768) or c for c in chunks
    ]
    embedder.embed_query.return_value = [0.1] * 768
    return embedder


@pytest.fixture
def mock_llm():
    """LLM mockado com resposta fixa e disponibilidade True."""
    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.return_value = "Ramon é Engenheiro de IA com experiência em RAG."
    return llm