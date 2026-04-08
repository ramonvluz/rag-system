# rag_system/tests/unit/conftest.py
"""Fixtures compartilhadas pelos testes unitários.

Todos os dados são sintéticos e não dependem de nenhum serviço externo.
Os objetos aqui definidos servem como entrada padrão para testes que
precisam de Chunks e Documents prontos sem acionar parsers ou LLMs.
"""
import pytest
from rag_system.core.models import Chunk, Document


@pytest.fixture
def sample_query():
    """Query sintética representando uma pergunta real ao sistema RAG."""
    return "Qual é a experiência profissional de Ramon?"


@pytest.fixture
def sample_chunks():
    """Dois Chunks mínimos sem embedding — usados em testes de busca e pipeline."""
    return [
        Chunk(
            chunk_id="doc1_chunk_0000",
            doc_id="doc1",
            text="Ramon trabalhou como Coordenador de Marketing na Studio 235.",
            metadata={"source": "cv.pdf", "filename": "cv.pdf"},
        ),
        Chunk(
            chunk_id="doc1_chunk_0001",
            doc_id="doc1",
            text="Experiência com Python e Machine Learning.",
            metadata={"source": "cv.pdf", "filename": "cv.pdf"},
        ),
    ]


@pytest.fixture
def sample_document():
    """Document sintético representando um PDF simples pós-parse."""
    return Document(
        doc_id="testdocabc12345",
        source_uri="/data/raw/test.pdf",
        text="Texto de exemplo para testes unitários.",
        metadata={"filename": "test.pdf", "filetype": "pdf"},
    )