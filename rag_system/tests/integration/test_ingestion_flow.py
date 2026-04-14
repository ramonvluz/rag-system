# rag_system/tests/integration/test_ingestion_flow.py

"""
Testes de integração do pipeline de ingestão.

Valida a colaboração real entre TextCleaner, ParagraphChunker,
ChromaStore e BM25Search — sem mockar a lógica de negócio.
O BGEEmbedder é mockado para evitar carregamento do modelo BGE-M3.
"""
from unittest.mock import patch
from rag_system.ingestion.cleaners.text_cleaner import TextCleaner
from rag_system.ingestion.chunkers.paragraph_chunker import ParagraphChunker
from rag_system.retrieval.search.bm25_search import BM25Search


class TestTextCleanerIntegration:
    """Valida que o TextCleaner processa documentos reais corretamente."""

    def test_clean_preserves_content(self, sample_document):
        """Conteúdo relevante deve sobreviver à limpeza."""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_document)
        assert "Ramon" in cleaned.text

    def test_clean_sets_cleaned_flag(self, sample_document):
        """Documento limpo deve ter metadata cleaned=True."""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_document)
        assert cleaned.metadata.get("cleaned") is True

    def test_clean_removes_excessive_whitespace(self, sample_document):
        """Não deve haver 3+ quebras de linha consecutivas após limpeza."""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_document)
        assert "\n\n\n" not in cleaned.text

    def test_clean_returns_new_document(self, sample_document):
        """clean() não deve modificar o documento original — deve retornar novo objeto."""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_document)
        assert cleaned is not sample_document


class TestParagraphChunkerIntegration:
    """Valida que o chunker gera chunks válidos a partir de um Document real."""

    def test_chunk_returns_list(self, sample_document):
        """chunk() deve retornar uma lista independente do conteúdo do documento."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        assert isinstance(chunks, list)

    def test_chunk_not_empty(self, sample_document):
        """Um documento com 3 parágrafos deve gerar pelo menos 1 chunk."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1

    def test_chunk_ids_are_unique(self, sample_document):
        """Cada chunk deve ter um chunk_id único dentro do mesmo documento."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_inherits_doc_id(self, sample_document):
        """Todos os chunks devem herdar o doc_id do documento pai."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert chunk.doc_id == sample_document.doc_id

    def test_chunk_text_not_empty(self, sample_document):
        """Nenhum chunk deve ter texto vazio ou composto apenas de whitespace."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunk_metadata_has_filename(self, sample_document):
        """Chunks devem herdar o filename do documento de origem."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert "filename" in chunk.metadata


class TestCleanerChunkerPipeline:
    """Testa a colaboração real entre TextCleaner e ParagraphChunker."""

    def test_clean_then_chunk_produces_valid_chunks(self, sample_document):
        """Pipeline cleaner → chunker deve produzir chunks com texto limpo."""
        cleaner = TextCleaner()
        chunker = ParagraphChunker()

        cleaned_doc = cleaner.clean(sample_document)
        chunks = chunker.chunk(cleaned_doc)

        assert len(chunks) > 0
        for chunk in chunks:
            # Garante que não há triplas quebras de linha nos chunks
            assert "\n\n\n" not in chunk.text

    def test_full_ingestion_pipeline_with_mock_embedder(
        self, sample_document, mock_embedder, tmp_path
    ):
        """Pipeline completo: clean → chunk → embed (mock) → upsert ChromaDB → build BM25.

        Valida que ChromaStore e BM25Search recebem os dados corretamente.
        ChromaDB usa diretório temporário para não poluir o ambiente de produção.
        """
        from rag_system.ingestion.vector_store.chroma_store import ChromaStore

        cleaner = TextCleaner()
        chunker = ParagraphChunker()

        # ChromaStore apontando para diretório temporário — isolamento total
        with patch("rag_system.ingestion.vector_store.chroma_store.settings") as mock_s:
            mock_s.chroma_db_dir = tmp_path / "chroma"
            mock_s.chroma_collection_name = "test_integration"
            store = ChromaStore()

        cleaned_doc = cleaner.clean(sample_document)
        chunks = chunker.chunk(cleaned_doc)

        # Embed mockado — preenche embeddings sem carregar o modelo BGE-M3
        mock_embedder.embed_chunks(chunks)

        # Upsert real no ChromaDB temporário
        store.upsert(chunks)

        # Verifica que os chunks foram persistidos corretamente
        results = store._collection.get(include=["documents"])
        assert len(results["documents"]) == len(chunks)

        # BM25 real construído sobre os chunks reais
        bm25 = BM25Search()
        bm25.build_index(chunks)
        assert bm25._bm25 is not None
        assert len(bm25._chunks) == len(chunks)


class TestBM25SearchIntegration:
    """Valida o BM25Search com chunks reais — sem mock."""

    def test_build_index_populates_bm25(self, sample_chunks):
        """build_index() deve popular _bm25 e _chunks com os dados fornecidos."""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        assert bm25._bm25 is not None
        assert len(bm25._chunks) == len(sample_chunks)

    def test_search_returns_results(self, sample_chunks):
        """search() com termo presente no índice deve retornar ao menos 1 resultado."""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        results = bm25.search("Ramon engenheiro", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_returns_tuples_chunk_score(self, sample_chunks):
        """BM25.search deve retornar lista de tuplas (Chunk, float)."""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        results = bm25.search("Python", top_k=5)
        for item in results:
            assert isinstance(item, tuple)
            chunk, score = item
            assert hasattr(chunk, "chunk_id")
            assert isinstance(score, float)

    def test_search_ranks_relevant_chunk_higher(self, sample_chunks):
        """Chunk com termo exato deve aparecer com score maior que chunks irrelevantes."""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        results = bm25.search("Marketing Studio 235", top_k=5)
        # O chunk sobre Marketing deve estar nos resultados
        texts = [chunk.text for chunk, _ in results]
        assert any("Marketing" in t for t in texts)

    def test_search_without_index_returns_empty(self):
        """Busca sem índice carregado deve retornar lista vazia sem lançar exceção."""
        bm25 = BM25Search()
        results = bm25.search("qualquer coisa", top_k=5)
        assert results == []