"""Classe base concreta para todos os chunkers do RAG System.

Fornece helpers compartilhados de geração de chunk_id e metadados,
eliminando duplicação entre ParagraphChunker e TableChunker.
"""

from rag_system.core.interfaces import BaseChunker
from rag_system.core.models import Document, Chunk
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class ChunkerBase(BaseChunker):
    """Classe base concreta para chunkers — adiciona helpers compartilhados.

    Subclasses devem obrigatoriamente implementar chunk(). Os métodos
    _build_chunk_id() e _build_chunk_metadata() são helpers prontos
    para uso, garantindo consistência de formato entre todos os chunkers.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide um Document em Chunks.

        Args:
            document: Document limpo, saída do TextCleaner.

        Returns:
            Lista de Chunks com chunk_id determinístico e metadados.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar chunk()")

    def _build_chunk_id(self, doc_id: str, index: int) -> str:
        """Gera chunk_id determinístico a partir do doc_id e posição.

        O formato ``{doc_id}_chunk_{index:04d}`` garante ordenação
        lexicográfica correta e torna o upsert no ChromaDB idempotente
        — re-indexar o mesmo documento não cria duplicatas.

        Args:
            doc_id: Identificador do documento de origem.
            index: Posição do chunk no documento (base 0).

        Returns:
            String no formato ``{doc_id}_chunk_{index:04d}``.
            Exemplo: ``relatorio_2024_a3f82c11_chunk_0003``
        """
        return f"{doc_id}_chunk_{index:04d}"

    def _build_chunk_metadata(self, document: Document, index: int, total: int) -> dict:
        """Monta metadados do chunk herdando os metadados do documento.

        Adiciona informações de posição (chunk_index, chunk_total) e
        rastreabilidade (doc_id) aos metadados herdados do Document.
        Esses campos são armazenados no ChromaDB junto ao chunk e
        permitem filtrar, ordenar e rastrear chunks por documento.

        Args:
            document: Document de origem do chunk.
            index: Posição do chunk no documento (base 0).
            total: Total de chunks gerados para o documento.
                Pode ser 0 durante a geração e atualizado depois.

        Returns:
            Dicionário com metadados do documento mesclados com
            chunk_index, chunk_total e doc_id.
        """
        return {
            **document.metadata,   # Herda source_uri, filename, file_type, etc.
            "chunk_index": index,
            "chunk_total": total,
            "doc_id": document.doc_id,
        }