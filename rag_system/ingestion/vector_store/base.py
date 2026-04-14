# rag_system/ingestion/vector_store/base.py

"""Classe base concreta para todas as implementações de vector store.

Segue o mesmo padrão de ParserBase e ChunkerBase — camada intermediária
entre a ABC (BaseVectorStore) e as implementações concretas (ChromaStore),
permitindo adicionar helpers compartilhados sem modificar a interface.
"""

from rag_system.core.interfaces import BaseVectorStore
from rag_system.core.models import Chunk


class VectorStoreBase(BaseVectorStore):
    """Classe base concreta para vector stores — pronta para helpers compartilhados.

    Atualmente serve como camada de indireção entre BaseVectorStore e
    ChromaStore. Helpers comuns a múltiplas implementações de vector store
    devem ser adicionados aqui — ex: validação de chunks antes do upsert,
    normalização de metadados, etc.
    """

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insere ou atualiza chunks no banco vetorial.

        Args:
            chunks: Lista de Chunks com embedding preenchido.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar upsert()")

    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """Busca os chunks mais similares ao embedding da query.

        Args:
            query_embedding: Vetor de embedding da query.
            top_k: Número máximo de chunks a retornar.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar search()")

    def delete(self, doc_id: str) -> None:
        """Remove todos os chunks de um documento.

        Args:
            doc_id: Identificador do documento cujos chunks serão removidos.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar delete()")