"""Reranker de chunks usando cross-encoder BGE (BAAI/bge-reranker-base).

Aplica uma etapa de reordenação mais precisa após a busca híbrida,
usando um cross-encoder que avalia cada par (query, chunk) de forma
conjunta — ao contrário dos bi-encoders (BGE-M3) que avaliam query
e chunk separadamente.

O custo computacional maior justifica aplicar o reranker apenas no
top-k da busca híbrida (padrão: 20 chunks), reduzindo para o top-k
final (padrão: 5 chunks) com maior precisão de relevância.
"""

from sentence_transformers import CrossEncoder

from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """Reranker baseado em cross-encoder BGE via sentence-transformers.

    Diferença fundamental em relação ao bi-encoder (BGE-M3):
        - Bi-encoder: embedding(query) vs embedding(chunk) → similaridade
        - Cross-encoder: score(query + chunk concatenados) → relevância real

    O cross-encoder produz scores de relevância mais precisos porque
    processa a interação entre query e chunk diretamente, mas é mais
    lento — por isso é aplicado apenas no top-k da busca híbrida.
    """

    def __init__(self) -> None:
        """Carrega o modelo cross-encoder a partir de settings.

        O modelo é carregado uma única vez na inicialização do pipeline
        e reutilizado para todas as chamadas de rerank().
        """
        logger.info(f"Carregando modelo de reranking: {settings.reranker_model}")
        self._model = CrossEncoder(settings.reranker_model)
        logger.info("Modelo de reranking carregado com sucesso.")

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Reordena chunks por relevância usando cross-encoder.

        Monta pares (query, chunk_text) e submete ao cross-encoder
        para scoring conjunto. Retorna apenas os top reranker_top_k
        chunks com maior score de relevância.

        Args:
            query: Pergunta original do usuário.
            chunks: Candidatos da busca híbrida (tipicamente top-20).

        Returns:
            Lista de até reranker_top_k Chunks reordenados por
            relevância real (maior score primeiro). Retorna lista
            vazia se chunks estiver vazio.
        """
        if not chunks:
            return []

        logger.info(f"Reranking de {len(chunks)} chunks para query: '{query}'")

        # Cross-encoder avalia cada par (query, chunk) de forma conjunta
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Ordena por score decrescente — maior score = maior relevância
        scored_chunks = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_chunks = [chunk for chunk, _ in scored_chunks[:settings.reranker_top_k]]
        logger.info(f"Reranking concluído: top-{len(top_chunks)} chunks selecionados.")
        return top_chunks
