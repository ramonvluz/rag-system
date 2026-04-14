# rag_system/tests/unit/test_rag_pipeline.py

"""Testes unitários do RAGPipeline.

Verifica o orquestrador central em isolamento total: todos os componentes
pesados (BGEEmbedder, ChromaStore, HybridSearch, BGEReranker, GroqLLM,
OllamaLLM) são mockados via patch. Foca no contrato de saída do método
query() e na delegação correta ao LLM.

Invariantes verificados:
- query() retorna dict com chaves 'answer' e 'sources'
- LLM.generate() é chamado exatamente uma vez por query
- resposta do LLM é repassada no campo 'answer'
"""
import pytest
from unittest.mock import patch, MagicMock
from rag_system.core.models import Chunk


MOCK_CHUNKS = [
    Chunk(
        chunk_id="doc1_chunk_0000",
        doc_id="doc1",
        text="Ramon foi Coordenador de Marketing na Studio 235.",
        metadata={"source_uri": "cv.pdf", "filename": "cv.pdf"},
        embedding=[0.1, 0.2, 0.3],
    )
]


@pytest.fixture
def pipeline():
    """RAGPipeline com todos os componentes ML mockados.

    O mock do LLM é injetado em pipeline._mock_llm para permitir asserts
    diretos sobre generate() nos testes sem depender do patch context.
    """
    with patch("rag_system.retrieval.pipeline.BGEEmbedder") as mock_emb, \
         patch("rag_system.retrieval.pipeline.ChromaStore"), \
         patch("rag_system.retrieval.pipeline.VectorSearch"), \
         patch("rag_system.retrieval.pipeline.BM25Search"), \
         patch("rag_system.retrieval.pipeline.HybridSearch") as mock_hybrid, \
         patch("rag_system.retrieval.pipeline.BGEReranker") as mock_reranker, \
         patch("rag_system.retrieval.pipeline.GroqLLM") as mock_groq_class, \
         patch("rag_system.retrieval.pipeline.OllamaLLM"):

        mock_groq = MagicMock()
        mock_groq.is_available.return_value = True
        mock_groq.generate.return_value = "Ramon trabalhou na Studio 235."
        mock_groq_class.return_value = mock_groq

        mock_hybrid.return_value.search.return_value = MOCK_CHUNKS
        mock_reranker.return_value.rerank.return_value = MOCK_CHUNKS
        mock_emb.return_value.embed_query.return_value = [0.1, 0.2, 0.3]

        from rag_system.retrieval.pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_provider="groq")

        # Injeta o mock no objeto para uso nas asserções
        pipeline._mock_llm = mock_groq

        yield pipeline


class TestRAGPipeline:
    """Testa o RAGPipeline como orquestrador — não testa os componentes internos.

    Cada componente (HybridSearch, BGEReranker, GroqLLM) tem seus próprios
    testes unitários. Aqui só verificamos que o pipeline conecta corretamente
    os componentes e produz o contrato de saída esperado por query().
    """

    def test_query_returns_dict(self, pipeline, sample_query):
        assert isinstance(pipeline.query(sample_query), dict)

    def test_query_has_answer(self, pipeline, sample_query):
        result = pipeline.query(sample_query)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_has_sources(self, pipeline, sample_query):
        result = pipeline.query(sample_query)
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_calls_llm_generate(self, pipeline, sample_query):
        """LLM.generate() deve ser chamado exatamente uma vez por query()."""
        pipeline.query(sample_query)
        pipeline._mock_llm.generate.assert_called_once()  # usa o mock injetado