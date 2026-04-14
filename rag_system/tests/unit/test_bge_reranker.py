# rag_system/tests/unit/test_bge_reranker.py

"""Testes unitários do BGEReranker.

Verifica o reranking cross-encoder sem carregar o modelo real. O CrossEncoder
é mockado, permitindo controlar os scores retornados e verificar ordenação,
truncamento (top-k) e comportamento com edge cases.

Invariantes verificados:
- lista vazia retorna lista vazia sem exceção
- resultado é ordenado por score decrescente
- retorna no máximo reranker_topk chunks
- predict é chamado exatamente uma vez por rerank()
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rag_system.core.models import Chunk
from rag_system.retrieval.reranker.bge_reranker import BGEReranker


def make_chunk(idx: int, text: str) -> Chunk:
    """Cria um Chunk sintético com embedding mínimo para uso nos testes."""
    return Chunk(
        chunk_id=f"doc1_chunk{idx:04d}",
        doc_id="doc1",
        text=text,
        metadata={"filename": "cv.pdf", "source_uri": "cv.pdf"},
        embedding=[0.1] * 3,
    )


@pytest.fixture
def reranker():
    """BGEReranker com CrossEncoder mockado e top-k fixo em 3.

    O mock é injetado diretamente em instance.model para permitir asserts
    diretos sobre predict() nos testes sem depender do patch context.
    """
    with patch("rag_system.retrieval.reranker.bge_reranker.CrossEncoder") as MockCE, \
         patch("rag_system.retrieval.reranker.bge_reranker.settings") as mock_settings:
        mock_settings.reranker_model = "BAAI/bge-reranker-base"
        mock_settings.reranker_topk = 3
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        instance = BGEReranker()
        instance.model = mock_model  # injeta mock para asserts
        yield instance, mock_settings


class TestBGEReranker:
    """Testa o BGEReranker em isolamento total.

    CrossEncoder é mockado para controlar scores e verificar a lógica de
    ordenação e truncamento sem dependência de GPU ou download de modelos.
    Testa contratos de tipo, edge cases (lista vazia, chunk único) e a
    delegação correta dos pares (query, chunk_text) ao predict().
    """

    def test_instantiation(self, reranker):
        instance, _ = reranker
        assert instance is not None

    def test_rerank_empty_chunks_returns_empty(self, reranker):
        instance, _ = reranker
        result = instance.rerank("Quem é Ramon?", [])
        assert result == []

    def test_rerank_returns_list(self, reranker):
        instance, _ = reranker
        chunks = [make_chunk(i, f"texto {i}") for i in range(5)]
        instance.model.predict.return_value = np.array([0.9, 0.2, 0.7, 0.4, 0.1])
        result = instance.rerank("Quem é Ramon?", chunks)
        assert isinstance(result, list)

    def test_rerank_respects_topk(self, reranker):
        """Resultado deve conter no máximo reranker_topk chunks."""
        instance, mock_settings = reranker
        mock_settings.reranker_topk = 3
        chunks = [make_chunk(i, f"texto {i}") for i in range(5)]
        instance.model.predict.return_value = np.array([0.9, 0.2, 0.7, 0.4, 0.1])
        result = instance.rerank("query", chunks)
        assert len(result) <= 3

    def test_rerank_orders_by_score_descending(self, reranker):
        """Chunk com maior score deve aparecer primeiro no resultado."""
        instance, _ = reranker
        chunks = [
            make_chunk(0, "texto irrelevante"),
            make_chunk(1, "texto muito relevante"),
            make_chunk(2, "texto médio"),
        ]
        # chunk índice 1 tem score mais alto → deve ser o primeiro
        instance.model.predict.return_value = np.array([0.1, 0.9, 0.5])
        result = instance.rerank("query relevante", chunks)
        assert result[0].chunk_id == "doc1_chunk0001"

    def test_rerank_calls_predict_with_correct_pairs(self, reranker):
        """predict() deve receber pares [query, chunk_text] para cada chunk."""
        instance, _ = reranker
        query = "Qual a experiência de Ramon?"
        chunks = [make_chunk(0, "texto a"), make_chunk(1, "texto b")]
        instance.model.predict.return_value = np.array([0.8, 0.3])
        instance.rerank(query, chunks)
        call_args = instance.model.predict.call_args[0][0]
        assert call_args[0] == [query, "texto a"]
        assert call_args[1] == [query, "texto b"]

    def test_rerank_result_contains_chunk_instances(self, reranker):
        instance, _ = reranker
        chunks = [make_chunk(i, f"texto {i}") for i in range(3)]
        instance.model.predict.return_value = np.array([0.9, 0.2, 0.7])
        result = instance.rerank("query", chunks)
        for item in result:
            assert isinstance(item, Chunk)

    def test_rerank_single_chunk_returns_it(self, reranker):
        """Lista com um único chunk deve retornar esse chunk independente do score."""
        instance, mock_settings = reranker
        mock_settings.reranker_topk = 3
        chunk = make_chunk(0, "único chunk disponível")
        instance.model.predict.return_value = np.array([0.85])
        result = instance.rerank("query", [chunk])
        assert len(result) == 1
        assert result[0].chunk_id == chunk.chunk_id

    def test_rerank_predict_called_once(self, reranker):
        """predict() deve ser chamado exatamente uma vez por chamada a rerank()."""
        instance, _ = reranker
        chunks = [make_chunk(i, f"texto {i}") for i in range(4)]
        instance.model.predict.return_value = np.array([0.5, 0.3, 0.8, 0.1])
        instance.rerank("query", chunks)
        instance.model.predict.assert_called_once()