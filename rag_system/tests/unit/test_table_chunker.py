# rag_system/tests/unit/test_table_chunker.py
"""Testes unitários do TableChunker.

Verifica a estratégia de chunking por linha para documentos tabulares.
Não há dependências externas — TableChunker opera puramente sobre strings
TSV/Markdown, sem models ou I/O.

Invariantes verificados:
- chunk_ids são únicos e começam com doc_id
- cada chunk herda o doc_id e os metadados do Document de origem
- texto vazio não lança exceção
- conteúdo relevante é preservado nos chunks gerados
"""

from rag_system.core.models import Document, Chunk
from rag_system.ingestion.chunkers.table_chunker import TableChunker


def make_table_doc(text: str = None) -> Document:
    """Document tabular sintético com 4 linhas de dados em formato TSV."""
    if text is None:
        text = (
            "nome\tidade\tcargo\n"
            "Ramon Valgas Luz\t30\tEngenheiro de IA\n"
            "João Silva\t25\tDesenvolvedor Backend\n"
            "Maria Costa\t35\tArquiteta de Software\n"
            "Pedro Alves\t28\tData Scientist\n"
        )
    return Document(
        doc_id="tabdoc_abc1234",
        source_uri="data/raw/tabela.csv",
        text=text,
        metadata={"filename": "tabela.csv", "filetype": "csv"},
    )


class TestTableChunker:
    """Testa o TableChunker com tabelas TSV sintéticas.

    Cada linha de dados deve virar um chunk independente. Testa contratos
    de tipo, unicidade de IDs, integridade de metadados e comportamento
    com inputs degenerados (texto vazio, tabela de uma linha só).
    """

    def test_chunk_returns_list(self):
        result = TableChunker().chunk(make_table_doc())
        assert isinstance(result, list)

    def test_chunk_not_empty(self):
        result = TableChunker().chunk(make_table_doc())
        assert len(result) > 0

    def test_chunk_ids_are_unique(self):
        result = TableChunker().chunk(make_table_doc())
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))

    def test_chunk_inherits_doc_id(self):
        doc = make_table_doc()
        for chunk in TableChunker().chunk(doc):
            assert chunk.doc_id == doc.doc_id

    def test_chunk_text_not_empty(self):
        for chunk in TableChunker().chunk(make_table_doc()):
            assert len(chunk.text.strip()) > 0

    def test_chunk_metadata_has_filename(self):
        for chunk in TableChunker().chunk(make_table_doc()):
            assert "filename" in chunk.metadata

    def test_chunk_ids_start_with_doc_id(self):
        doc = make_table_doc()
        for chunk in TableChunker().chunk(doc):
            assert chunk.chunk_id.startswith(doc.doc_id)

    def test_chunk_returns_chunk_instances(self):
        for chunk in TableChunker().chunk(make_table_doc()):
            assert isinstance(chunk, Chunk)

    def test_chunk_empty_text_does_not_raise(self):
        """Texto vazio não deve lançar exceção — retorna lista vazia."""
        result = TableChunker().chunk(make_table_doc(text=""))
        assert isinstance(result, list)

    def test_chunk_single_row_document(self):
        """Tabela com uma única linha de dados deve gerar ao menos 1 chunk."""
        doc = make_table_doc(text="nome\tidade\nRamon\t30\n")
        result = TableChunker().chunk(doc)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_preserves_content(self):
        """Conteúdo relevante do documento deve estar presente nos chunks gerados."""
        doc = make_table_doc()
        all_text = " ".join(c.text for c in TableChunker().chunk(doc))
        assert "Ramon" in all_text