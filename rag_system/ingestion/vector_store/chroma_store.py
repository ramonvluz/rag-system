"""Implementação do banco vetorial usando ChromaDB.

Persiste chunks com embeddings em disco via ChromaDB PersistentClient,
usando similaridade de cosseno como métrica de distância — compatível
com os embeddings L2-normalizados gerados pelo BGEEmbedder.
"""

import chromadb

from rag_system.ingestion.vector_store.base import VectorStoreBase
from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class ChromaStore(VectorStoreBase):
    """Implementação de VectorStoreBase usando ChromaDB persistente.

    Armazena chunks em disco no diretório definido por settings.chroma_db_dir.
    Usa o algoritmo HNSW com espaço de cosseno, compatível com embeddings
    L2-normalizados do BGE-M3.

    O upsert é idempotente — re-indexar o mesmo documento substitui
    os chunks existentes pelo chunk_id, sem criar duplicatas.
    """

    def __init__(self) -> None:
        """Conecta ao ChromaDB persistente e obtém ou cria a coleção.

        Cria o diretório chroma_db_dir se não existir. A coleção é
        configurada com métrica de cosseno na criação — se já existir,
        a configuração original é mantida.
        """
        settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosseno — correto para embeddings L2-normalizados
        )
        logger.info(f"ChromaDB conectado — coleção: '{settings.chroma_collection_name}'")

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insere ou atualiza chunks no ChromaDB.

        Operação idempotente — chunks com chunk_id já existente são
        substituídos, não duplicados. Requer que todos os chunks
        tenham o campo embedding preenchido.

        Args:
            chunks: Lista de Chunks com embedding preenchido,
                saída do BGEEmbedder.
        """
        if not chunks:
            logger.warning("Nenhum chunk para upsert.")
            return

        # Desempacota os campos de cada chunk para o formato esperado pelo ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"{len(chunks)} chunks persistidos no ChromaDB.")

    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """Busca os chunks mais similares ao embedding da query.

        Args:
            query_embedding: Vetor L2-normalizado da query,
                gerado pelo BGEEmbedder.embed_query().
            top_k: Número máximo de chunks a retornar.

        Returns:
            Lista de até top_k Chunks ordenados por similaridade
            de cosseno (maior similaridade primeiro).
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Reconstrói objetos Chunk a partir do resultado flat do ChromaDB
        chunks = []
        for i, chunk_id in enumerate(results["ids"][0]):
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=results["metadatas"][0][i].get("doc_id", ""),
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            ))
        return chunks

    def delete(self, doc_id: str) -> None:
        """Remove todos os chunks de um documento do ChromaDB.

        Busca os chunks pelo filtro de metadados doc_id antes de deletar,
        logando um aviso se nenhum chunk for encontrado.

        Args:
            doc_id: Identificador do documento cujos chunks
                serão removidos.
        """
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"{len(results['ids'])} chunks deletados para doc_id: '{doc_id}'")
        else:
            logger.warning(f"Nenhum chunk encontrado para doc_id: '{doc_id}'")

    @property
    def count(self) -> int:
        """Retorna o número total de chunks armazenados na coleção."""
        return self._collection.count()