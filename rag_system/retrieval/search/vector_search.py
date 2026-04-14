# rag_system/retrieval/search/vector_search.py

"""Busca semântica por similaridade vetorial usando ChromaDB.

Responsável pela etapa de busca densa no pipeline híbrido —
converte a query em embedding e recupera os chunks mais similares
no espaço vetorial do ChromaDB.
"""

from rag_system.ingestion.vector_store.chroma_store import ChromaStore
from rag_system.ingestion.embedders.bge_embedder import BGEEmbedder
from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class VectorSearch:
    """Busca semântica por similaridade de cosseno no ChromaDB.

    Etapa densa do pipeline de busca híbrida — excelente para
    similaridade de significado, mas menos eficaz para termos
    exatos como siglas e nomes próprios (complementada pelo BM25).
    """

    def __init__(self, embedder: BGEEmbedder, store: ChromaStore) -> None:
        """Inicializa a busca vetorial com embedder e store compartilhados.

        Args:
            embedder: Instância do BGEEmbedder para gerar o embedding da query.
            store: Instância do ChromaStore para executar a busca.
        """
        self._embedder = embedder
        self._store = store

    def search(self, query: str) -> list[Chunk]:
        """Executa busca semântica e retorna os chunks mais similares.

        Converte a query em embedding com o mesmo modelo usado na
        indexação (BGE-M3) para garantir compatibilidade no espaço
        vetorial, e delega a busca ao ChromaStore.

        Args:
            query: Pergunta ou texto de busca do usuário.

        Returns:
            Lista de até vector_search_top_k Chunks ordenados por
            similaridade de cosseno (maior similaridade primeiro).
        """
        logger.info(f"Busca vetorial: '{query}'")
        query_embedding = self._embedder.embed_query(query)
        results = self._store.search(query_embedding, top_k=settings.vector_search_top_k)
        logger.info(f"{len(results)} chunks recuperados via busca vetorial.")
        return results