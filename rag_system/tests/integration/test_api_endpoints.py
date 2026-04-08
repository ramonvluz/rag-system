# rag_system/tests/integration/test_api_endpoints.py
"""
Testes de integração dos endpoints FastAPI.

Usa o TestClient do FastAPI para simular requisições HTTP reais contra
a aplicação, validando contratos de entrada/saída, códigos HTTP e
tratamento de erros — sem subir um servidor real.
Todos os componentes pesados (pipeline, embedder) são mockados.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    TestClient com RAGPipeline e BGEEmbedder completamente mockados.
    Reseta os singletons das rotas antes de cada teste para garantir
    isolamento total entre chamadas.
    """
    with patch("rag_system.retrieval.pipeline.BGEEmbedder"), \
         patch("rag_system.retrieval.pipeline.ChromaStore"), \
         patch("rag_system.retrieval.pipeline.VectorSearch"), \
         patch("rag_system.retrieval.pipeline.BM25Search"), \
         patch("rag_system.retrieval.pipeline.HybridSearch") as MockHybrid, \
         patch("rag_system.retrieval.pipeline.BGEReranker") as MockReranker, \
         patch("rag_system.retrieval.pipeline.GroqLLM") as MockGroq, \
         patch("rag_system.retrieval.pipeline.OllamaLLM"), \
         patch("rag_system.api.routes.ingest.BGEEmbedder"), \
         patch("rag_system.api.routes.ingest.ChromaStore"), \
         patch("rag_system.api.routes.ingest.BM25Search"):

        from rag_system.core.models import Chunk

        mock_chunk = Chunk(
            chunk_id="test_chunk_0000",
            doc_id="test_doc",
            text="Ramon é Engenheiro de IA.",
            metadata={"filename": "test.pdf", "source_uri": "test.pdf"},
            embedding=[0.1] * 768,
        )

        MockHybrid.return_value.search.return_value = [mock_chunk]
        MockReranker.return_value.rerank.return_value = [mock_chunk]

        mock_groq = MagicMock()
        mock_groq.is_available.return_value = True
        mock_groq.generate.return_value = "Ramon é Engenheiro de IA."
        MockGroq.return_value = mock_groq

        # Reseta singletons das rotas para isolamento entre testes
        import rag_system.api.routes.query as query_route
        import rag_system.api.routes.ingest as ingest_route
        query_route.pipeline = None
        ingest_route.embedder = None
        ingest_route.store = None

        from rag_system.api.main import app
        yield TestClient(app)


class TestHealthEndpoint:
    """Testa o endpoint GET /health.

    Verifica que a API está no ar e retorna o contrato mínimo esperado:
    status 200, campo 'status' == 'ok' e campo 'version' presente.
    """

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


class TestQueryEndpoint:
    """Testa o endpoint POST /query.

    Cobre o caminho feliz (pergunta válida → 200 com 'answer' e 'sources')
    e os casos de erro de validação Pydantic (pergunta curta demais,
    campo ausente, string vazia → 422).
    """

    def test_query_returns_200(self, client):
        response = client.post("/query", json={"question": "Quem é Ramon?"})
        assert response.status_code == 200

    def test_query_response_has_answer(self, client):
        data = client.post("/query", json={"question": "Quem é Ramon?"}).json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_query_response_has_sources(self, client):
        data = client.post("/query", json={"question": "Quem é Ramon?"}).json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_too_short_returns_422(self, client):
        """Pergunta com menos de 3 caracteres deve ser rejeitada pelo Pydantic."""
        response = client.post("/query", json={"question": "Oi"})
        assert response.status_code == 422

    def test_query_missing_field_returns_422(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_empty_string_returns_422(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422


class TestIngestEndpoint:
    """Testa o endpoint POST /ingest.

    Cobre: arquivo inexistente (404), arquivo real em disco com parsers
    mockados (200), payload sem filepath (422). Garante que o contrato
    de saída inclui status == 'success' em caso de sucesso.
    """

    def test_ingest_file_not_found_returns_404(self, client):
        response = client.post(
            "/ingest",
            json={"filepath": "/caminho/que/nao/existe.pdf"},
        )
        assert response.status_code == 404

    def test_ingest_existing_file_returns_200(self, client, tmp_path):
        """Arquivo real em disco deve ser ingerido com sucesso (parsers mockados)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Conteúdo de teste para ingestão.")

        with patch("rag_system.api.routes.ingest.get_parser") as mock_factory, \
             patch("rag_system.api.routes.ingest.TextCleaner") as MockCleaner, \
             patch("rag_system.api.routes.ingest.ParagraphChunker") as MockChunker, \
             patch("builtins.open", mock_open()), \
             patch("pathlib.Path.mkdir"):

            from rag_system.core.models import Document, Chunk

            mock_doc = Document(
                doc_id="test_doc_abc",
                source_uri=str(test_file),
                text="Conteúdo de teste.",
                metadata={"filename": "test.txt"},
            )
            mock_chunk = Chunk(
                chunk_id="test_doc_abc_chunk_0000",
                doc_id="test_doc_abc",
                text="Conteúdo de teste.",
                metadata={"filename": "test.txt"},
                embedding=[0.1] * 768,
            )
            mock_factory.return_value.parse.return_value = mock_doc
            MockCleaner.return_value.clean.return_value = mock_doc
            MockChunker.return_value.chunk.return_value = [mock_chunk]

            response = client.post("/ingest", json={"filepath": str(test_file)})

        assert response.status_code == 200

    def test_ingest_response_has_status(self, client, tmp_path):
        """Resposta de ingestão bem-sucedida deve ter campo status='success'."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("texto qualquer")

        with patch("rag_system.api.routes.ingest.get_parser") as mock_factory, \
             patch("rag_system.api.routes.ingest.TextCleaner") as MockCleaner, \
             patch("rag_system.api.routes.ingest.ParagraphChunker") as MockChunker, \
             patch("builtins.open", mock_open()), \
             patch("pathlib.Path.mkdir"):

            from rag_system.core.models import Document, Chunk

            mock_doc = Document(
                doc_id="doc_abc",
                source_uri=str(test_file),
                text="texto qualquer",
                metadata={"filename": "doc.txt"},
            )
            mock_chunk = Chunk(
                chunk_id="doc_abc_chunk_0000",
                doc_id="doc_abc",
                text="texto qualquer",
                metadata={"filename": "doc.txt"},
                embedding=[0.1] * 768,
            )
            mock_factory.return_value.parse.return_value = mock_doc
            MockCleaner.return_value.clean.return_value = mock_doc
            MockChunker.return_value.chunk.return_value = [mock_chunk]

            data = client.post("/ingest", json={"filepath": str(test_file)}).json()

        assert data["status"] == "success"

    def test_ingest_missing_filepath_returns_422(self, client):
        response = client.post("/ingest", json={})
        assert response.status_code == 422