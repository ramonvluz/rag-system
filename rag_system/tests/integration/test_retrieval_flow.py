# rag_system/tests/integration/test_retrieval_flow.py
"""
Testes de integração do pipeline de recuperação.

Valida a colaboração entre VectorSearch, BM25Search, HybridSearch
e PromptBuilder com dados reais no ChromaDB temporário.
BGEEmbedder, BGEReranker e LLM são mockados.
"""
import pytest
from unittest.mock import patch, MagicMock
from rag_system.retrieval.search.bm25_search import BM25Search
from rag_system.retrieval.generator.prompt_builder import PromptBuilder
from rag_system.core.models import Chunk


class TestHybridSearchIntegration:
    """Valida a fusão real entre VectorSearch e BM25Search.

    ChromaStore usa diretório temporário. Embedder é mockado.
    """

    @pytest.fixture
    def populated_stores(self, sample_chunks, tmp_path):
        """ChromaStore e BM25Search populados com sample_chunks.

        Três patches de settings são necessários porque cada componente
        lê settings de forma independente no momento da instanciação.
        """
        from rag_system.ingestion.vector_store.chroma_store import ChromaStore
        from rag_system.retrieval.search.vector_search import VectorSearch
        from rag_system.retrieval.search.hybrid_search import HybridSearch

        with patch("rag_system.ingestion.vector_store.chroma_store.settings") as ms, \
             patch("rag_system.retrieval.search.vector_search.settings") as ms2, \
             patch("rag_system.retrieval.search.hybrid_search.settings") as ms3:

            ms.chroma_db_dir = tmp_path / "chroma"
            ms.chroma_collection_name = "test_retrieval"
            ms2.chroma_db_dir = tmp_path / "chroma"
            ms2.chroma_collection_name = "test_retrieval"
            ms2.vector_search_top_k = 5
            ms3.vector_search_top_k = 5
            ms3.hybrid_semantic_weight = 0.7
            ms3.hybrid_bm25_weight = 0.3

            store = ChromaStore()
            store.upsert(sample_chunks)

            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 768

            vector_search = VectorSearch(embedder=mock_embedder, store=store)

            bm25 = BM25Search()
            bm25.build_index(sample_chunks)

            hybrid = HybridSearch(
                vector_search=vector_search,
                bm25_search=bm25,
            )
            yield hybrid

    def test_hybrid_search_returns_chunks(self, populated_stores):
        """search() deve retornar lista não-vazia de chunks para query válida."""
        results = populated_stores.search("experiência Ramon")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_results_are_chunk_instances(self, populated_stores):
        """Todos os itens do resultado devem ser instâncias de Chunk."""
        results = populated_stores.search("Python FastAPI")
        for item in results:
            assert isinstance(item, Chunk)

    def test_hybrid_no_duplicate_chunk_ids(self, populated_stores):
        """Fusão não deve gerar duplicatas mesmo com chunks em ambos os índices."""
        results = populated_stores.search("Ramon engenheiro IA")
        ids = [c.chunk_id for c in results]
        assert len(ids) == len(set(ids))

    def test_hybrid_finds_exact_term(self, populated_stores):
        """BM25 deve garantir que termos exatos como 'Studio 235' sejam encontrados."""
        results = populated_stores.search("Studio 235")
        texts = [c.text for c in results]
        assert any("Studio 235" in t for t in texts)


class TestPromptBuilderIntegration:
    """Valida o PromptBuilder com chunks reais — sem nenhum mock.

    Verifica que o prompt gerado contém a query e o texto dos chunks,
    que sources é deduplicado, e que lista vazia de chunks não lança exceção.
    """

    def test_build_returns_tuple(self, sample_chunks):
        """build() deve retornar uma tupla (prompt, sources)."""
        builder = PromptBuilder()
        result = builder.build("Quem é Ramon?", sample_chunks)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_build_prompt_is_string(self, sample_chunks):
        """Primeiro elemento da tupla deve ser string não-vazia."""
        builder = PromptBuilder()
        prompt, _ = builder.build("Quem é Ramon?", sample_chunks)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_sources_is_list(self, sample_chunks):
        """Segundo elemento da tupla deve ser lista."""
        builder = PromptBuilder()
        _, sources = builder.build("Quem é Ramon?", sample_chunks)
        assert isinstance(sources, list)

    def test_build_prompt_contains_query(self, sample_chunks):
        """O prompt gerado deve conter a query do usuário."""
        builder = PromptBuilder()
        query = "Qual é a experiência de Ramon?"
        prompt, _ = builder.build(query, sample_chunks)
        assert query in prompt

    def test_build_prompt_contains_chunk_text(self, sample_chunks):
        """O prompt deve injetar o texto dos chunks como contexto."""
        builder = PromptBuilder()
        prompt, _ = builder.build("pergunta", sample_chunks)
        assert sample_chunks[0].text in prompt

    def test_build_sources_deduplicates(self, sample_chunks):
        """Todos os chunks têm a mesma fonte — sources deve ter itens únicos."""
        builder = PromptBuilder()
        _, sources = builder.build("pergunta", sample_chunks)
        assert len(sources) == len(set(sources))

    def test_build_with_empty_chunks(self):
        """PromptBuilder com lista vazia não deve lançar exceção."""
        builder = PromptBuilder()
        prompt, sources = builder.build("pergunta", [])
        assert isinstance(prompt, str)
        assert isinstance(sources, list)


class TestRAGPipelineIntegration:
    """Testa o RAGPipeline completo com ChromaDB real (tmp) e LLM mockado.

    Valida o fluxo: query → hybrid search → rerank → prompt → generate.
    """

    @pytest.fixture
    def integrated_pipeline(self, sample_chunks, tmp_path):
        """Pipeline com ChromaDB + BM25 reais e modelos ML mockados.

        HybridSearch é injetado via patch para retornar sample_chunks reais,
        simulando recuperação sem depender de embeddings reais.
        """
        with patch("rag_system.retrieval.pipeline.BGEEmbedder") as MockEmb, \
             patch("rag_system.retrieval.pipeline.ChromaStore"), \
             patch("rag_system.retrieval.pipeline.VectorSearch"), \
             patch("rag_system.retrieval.pipeline.BM25Search"), \
             patch("rag_system.retrieval.pipeline.BGEReranker") as MockReranker, \
             patch("rag_system.retrieval.pipeline.GroqLLM") as MockGroq, \
             patch("rag_system.retrieval.pipeline.OllamaLLM"):

            MockEmb.return_value.embed_query.return_value = [0.1] * 768

            from rag_system.retrieval.search.hybrid_search import HybridSearch
            mock_hybrid = MagicMock(spec=HybridSearch)
            mock_hybrid.search.return_value = sample_chunks

            # Injeta HybridSearch mockado no pipeline
            with patch("rag_system.retrieval.pipeline.HybridSearch",
                       return_value=mock_hybrid):

                # Reranker devolve chunks sem modificar — transparente ao pipeline
                MockReranker.return_value.rerank.side_effect = (
                    lambda q, chunks: chunks
                )

                mock_groq = MagicMock()
                mock_groq.is_available.return_value = True
                mock_groq.generate.return_value = (
                    "Ramon é Engenheiro de IA com foco em sistemas RAG."
                )
                MockGroq.return_value = mock_groq

                from rag_system.retrieval.pipeline import RAGPipeline
                pipeline = RAGPipeline(llm_provider="groq")
                # Injeta mocks para asserts diretos nos testes
                pipeline._mock_llm = mock_groq
                pipeline._mock_hybrid = mock_hybrid
                yield pipeline

    def test_query_returns_dict(self, integrated_pipeline):
        """query() deve retornar dict com as chaves do contrato."""
        result = integrated_pipeline.query("Quem é Ramon?")
        assert isinstance(result, dict)

    def test_query_has_answer(self, integrated_pipeline):
        """Campo 'answer' deve estar presente e não-vazio."""
        result = integrated_pipeline.query("Quem é Ramon?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_has_sources(self, integrated_pipeline):
        """Campo 'sources' deve estar presente e ser uma lista."""
        result = integrated_pipeline.query("Quem é Ramon?")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_has_chunks(self, integrated_pipeline):
        """Campo 'chunks' deve estar presente e ser uma lista."""
        result = integrated_pipeline.query("Quem é Ramon?")
        assert "chunks" in result
        assert isinstance(result["chunks"], list)

    def test_query_calls_search_with_question(self, integrated_pipeline):
        """HybridSearch.search() deve receber exatamente a pergunta do usuário."""
        question = "Qual a experiência de Ramon?"
        integrated_pipeline.query(question)
        integrated_pipeline._mock_hybrid.search.assert_called_once_with(question)

    def test_query_calls_llm_generate(self, integrated_pipeline):
        """LLM.generate() deve ser chamado exatamente uma vez por query()."""
        integrated_pipeline.query("Quem é Ramon?")
        integrated_pipeline._mock_llm.generate.assert_called_once()

    def test_short_query_returns_early(self, integrated_pipeline):
        """Queries muito curtas devem retornar sem chamar o LLM."""
        result = integrated_pipeline.query("Oi")
        assert "answer" in result
        integrated_pipeline._mock_llm.generate.assert_not_called()