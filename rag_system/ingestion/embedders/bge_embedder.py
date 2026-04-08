"""Gerador de embeddings usando o modelo BGE-M3 (BAAI/bge-m3).

Utiliza sentence-transformers para gerar embeddings densos normalizados
(L2), compatíveis com busca por similaridade de cosseno no ChromaDB.
O mesmo modelo é usado para chunks (indexação) e queries (recuperação),
garantindo consistência no espaço vetorial.
"""

from sentence_transformers import SentenceTransformer

from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """Embedder baseado no modelo BGE-M3 via sentence-transformers.

    Responsável por duas operações distintas no pipeline:
        - ``embed_chunks()``: geração em batch durante a indexação
        - ``embed_query()``: geração individual durante a recuperação

    A normalização L2 é aplicada em ambos os casos — vetores normalizados
    permitem usar produto escalar como proxy para similaridade de cosseno,
    o que é mais eficiente computacionalmente no ChromaDB.
    """

    def __init__(self) -> None:
        """Carrega o modelo de embedding a partir de settings.

        O modelo é carregado uma única vez na inicialização e reutilizado
        para todas as chamadas subsequentes — evita recarregamento custoso
        entre chunks de um mesmo pipeline de ingestão.
        """
        logger.info(f"Carregando modelo de embedding: {settings.embedding_model}")
        self._model = SentenceTransformer(settings.embedding_model)
        logger.info("Modelo carregado com sucesso.")

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Gera embeddings em batch para uma lista de chunks.

        Modifica os chunks in-place preenchendo o campo ``embedding``
        de cada instância e retorna a mesma lista atualizada.

        Args:
            chunks: Lista de Chunks sem embedding (saída do chunker).

        Returns:
            A mesma lista de Chunks com o campo embedding preenchido
            como list[float], pronta para upsert no ChromaDB.
        """
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Gerando embeddings para {len(texts)} chunks (batch={settings.embedding_batch_size})")

        embeddings = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 norm — produto escalar ≡ similaridade de cosseno
        )

        # Preenche o campo embedding de cada chunk com lista Python nativa
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        logger.info("Embeddings gerados com sucesso.")
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Gera embedding para a query do usuário em tempo de recuperação.

        Aplica a mesma normalização L2 usada nos chunks para garantir
        compatibilidade no espaço vetorial durante a busca.

        Args:
            query: Texto da pergunta do usuário.

        Returns:
            Vetor de embedding como list[float], compatível com
            VectorSearch e HybridSearch.
        """
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,  # Deve ser idêntico ao usado em embed_chunks
        )
        return embedding.tolist()