# rag_system/tests/unit/test_bge_embedder.py

"""Testes unitários do BGEEmbedder.

Verifica embed_chunks (batch) e embed_query (individual) sem carregar o
modelo BGE-M3 real. O SentenceTransformer é mockado — execução rápida em
qualquer ambiente, sem GPU e sem download de pesos.

Invariantes verificados:
- embed_chunks modifica a lista original in-place e a retorna
- todos os chunks de um batch recebem embedding com dimensão idêntica
- embed_query retorna list[float], não ndarray
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag_system.core.models import Chunk
from rag_system.ingestion.embedders.bge_embedder import BGEEmbedder


def make_chunk(idx: int, text: str) -> Chunk:
    """Cria um Chunk sintético sem embedding para uso nos testes."""
    return Chunk(
        chunk_id=f"doc1_chunk_{idx:04d}",
        doc_id="doc1",
        text=text,
        metadata={},
    )


@pytest.fixture
def embedder():
    """BGEEmbedder com SentenceTransformer mockado retornando vetores 3-dim.

    O mock_model.encode retorna um array 2D (batch) por padrão.
    Testes que chamam embed_query precisam ajustar para 1D — veja
    test_embed_query_returns_list_of_floats.
    """
    with patch("rag_system.ingestion.embedders.bge_embedder.SentenceTransformer") as mock_st, \
         patch("rag_system.ingestion.embedders.bge_embedder.settings") as mock_settings:

        mock_settings.embedding_model = "BAAI/bge-m3"
        mock_settings.embedding_batch_size = 16

        mock_model = MagicMock()
        # Retorno padrão 2D: simula batch de 2 chunks com vetores 3-dim
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st.return_value = mock_model

        yield BGEEmbedder()


class TestBGEEmbedder:
    """Testa o BGEEmbedder em isolamento total.

    SentenceTransformer é mockado — nenhum modelo é carregado em disco.
    Foca em contratos: tipos de retorno, modificação in-place e consistência
    de dimensão entre embed_chunks e embed_query.
    """

    def test_instantiation(self, embedder):
        assert embedder is not None

    def test_embed_chunks_returns_list(self, embedder):
        chunks = [make_chunk(0, "texto 1"), make_chunk(1, "texto 2")]
        assert isinstance(embedder.embed_chunks(chunks), list)

    def test_embed_chunks_fills_embedding(self, embedder):
        chunks = [make_chunk(0, "texto 1"), make_chunk(1, "texto 2")]
        result = embedder.embed_chunks(chunks)
        for chunk in result:
            assert chunk.embedding is not None
            assert isinstance(chunk.embedding, list)

    def test_embed_chunks_same_count(self, embedder):
        chunks = [make_chunk(i, f"texto {i}") for i in range(2)]
        assert len(embedder.embed_chunks(chunks)) == 2

    def test_embed_chunks_modifies_in_place(self, embedder):
        """embed_chunks deve modificar a lista original e retorná-la — sem cópia."""
        chunks = [make_chunk(0, "texto")]
        result = embedder.embed_chunks(chunks)
        assert result is chunks

    def test_embed_query_returns_list_of_floats(self, embedder):
        # O mock_model já está configurado na fixture — só precisa ajustar o retorno para 1D
        embedder._model.encode.return_value = np.array([0.1, 0.2, 0.3])
        result = embedder.embed_query("qual a experiência de Ramon?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embedding_consistent_dimension(self, embedder):
        """Todos os chunks de um batch devem ter embedding de dimensão idêntica."""
        chunks = [make_chunk(i, f"texto {i}") for i in range(2)]
        result = embedder.embed_chunks(chunks)
        dims = [len(c.embedding) for c in result]
        assert len(set(dims)) == 1