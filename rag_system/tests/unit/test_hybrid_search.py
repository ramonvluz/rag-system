# rag_system/tests/unit/test_hybrid_search.py

"""Testes unitários do HybridSearch.

Verifica a fusão de scores semânticos e BM25 sem ChromaDB real.
VectorSearch e BM25Search são mockados, permitindo controlar os candidatos
retornados por cada motor e verificar que a fusão produz resultados corretos.

Invariantes verificados:
- ambos os motores são chamados exatamente uma vez por query
- resultado não contém chunk_ids duplicados (overlap entre motores é resolvido)
- cada item do resultado é uma instância de Chunk
"""
import pytest
from unittest.mock import MagicMock, patch
from rag_system.core.models import Chunk
from rag_system.retrieval.search.hybrid_search import HybridSearch


def make_chunk(idx: int, text: str) -> Chunk:
    """Cria um Chunk sintético com embedding mínimo para uso nos testes."""
    return Chunk(
        chunk_id=f"doc1_chunk_{idx:04d}",
        doc_id="doc1",
        text=text,
        metadata={"filename": "cv.pdf"},
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def mock_vector_search():
    """VectorSearch mockado retornando dois chunks semânticos."""
    vs = MagicMock()
    vs.search.return_value = [
        make_chunk(0, "Ramon é especialista em IA."),
        make_chunk(1, "Formação em Engenharia."),
    ]
    return vs


@pytest.fixture
def mock_bm25_search():
    """BM25Search mockado retornando chunks com scores lexicais.

    chunk_id=0 aparece em ambos os motores intencionalmente — permite
    verificar que a fusão elimina duplicatas corretamente.
    """
    bm = MagicMock()
    bm.bm25 = MagicMock()
    bm.chunks = [make_chunk(0, "Ramon é especialista em IA.")]
    bm.search.return_value = [
        (make_chunk(2, "Python e Machine Learning."), 8.5),
        (make_chunk(0, "Ramon é especialista em IA."), 7.2),
    ]
    return bm


@pytest.fixture
def hybrid(mock_vector_search, mock_bm25_search):
    """HybridSearch com ambos os motores mockados e pesos 0.7/0.3."""
    with patch("rag_system.retrieval.search.hybrid_search.settings") as mock_settings:
        mock_settings.vector_search_top_k = 5
        mock_settings.hybrid_semantic_weight = 0.7
        mock_settings.hybrid_bm25_weight = 0.3
        yield HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
        )


class TestHybridSearch:
    """Testa a lógica de fusão do HybridSearch em isolamento.

    Foca nos contratos de saída (tipos, ausência de duplicatas) e na
    colaboração correta com os dois motores de busca subjacentes.
    A qualidade do ranking é coberta pelos testes de integração.
    """

    def test_search_returns_list(self, hybrid, sample_query):
        assert isinstance(hybrid.search(sample_query), list)

    def test_search_not_empty(self, hybrid, sample_query):
        assert len(hybrid.search(sample_query)) > 0

    def test_calls_both_engines(self, hybrid, mock_vector_search, mock_bm25_search, sample_query):
        """Ambos os motores devem ser consultados exatamente uma vez por search()."""
        hybrid.search(sample_query)
        mock_vector_search.search.assert_called_once_with(sample_query)
        mock_bm25_search.search.assert_called_once()

    def test_results_are_chunks(self, hybrid, sample_query):
        for item in hybrid.search(sample_query):
            assert isinstance(item, Chunk)

    def test_no_duplicate_chunk_ids(self, hybrid, sample_query):
        """Fusão não deve introduzir duplicatas mesmo com overlap entre os motores."""
        results = hybrid.search(sample_query)
        ids = [c.chunk_id for c in results]
        assert len(ids) == len(set(ids))