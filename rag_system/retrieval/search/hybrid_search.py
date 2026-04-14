# rag_system/retrieval/search/hybrid_search.py

"""Busca híbrida — fusão ponderada de busca semântica e BM25.

Combina os resultados do VectorSearch (busca densa) e do BM25Search
(busca esparsa) em um único ranking, aproveitando os pontos fortes
de cada abordagem:

    - Vetorial: captura similaridade semântica e sinônimos
    - BM25: captura correspondência exata de termos, siglas e nomes próprios

O score final é calculado como:
    score = w_sem × score_vetorial + w_bm25 × score_bm25

Os pesos são configuráveis via .env (HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT).
"""

from rag_system.retrieval.search.vector_search import VectorSearch
from rag_system.retrieval.search.bm25_search import BM25Search
from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class HybridSearch:
    """Fusão ponderada de busca vetorial e BM25.

    Normaliza os scores de cada método para a escala [0, 1] antes
    da fusão, garantindo que nenhum método domine apenas por ter
    scores em escalas diferentes:

        - Vetorial: normalizado por posição — 1.0 para o 1º resultado,
          decrescente até 0.0 para o último
        - BM25: normalizado pelo score máximo do batch — score / max_score
    """

    def __init__(self, vector_search: VectorSearch, bm25_search: BM25Search) -> None:
        """Inicializa a busca híbrida com as duas estratégias de busca.

        Args:
            vector_search: Instância do VectorSearch para busca semântica.
            bm25_search: Instância do BM25Search para busca lexical.
        """
        self._vector_search = vector_search
        self._bm25_search = bm25_search

    def search(self, query: str) -> list[Chunk]:
        """Executa busca híbrida e retorna os chunks com maior score combinado.

        Fluxo:
            1. Executa VectorSearch e normaliza scores por posição
            2. Executa BM25Search e normaliza scores pelo máximo do batch
            3. Une os chunk_ids de ambos os resultados
            4. Calcula score híbrido ponderado para cada chunk
            5. Ordena por score decrescente e retorna top_k

        Args:
            query: Pergunta ou texto de busca do usuário.

        Returns:
            Lista de até vector_search_top_k Chunks ordenados pelo
            score híbrido ponderado (maior score primeiro).
        """
        top_k = settings.vector_search_top_k

        # --- Busca vetorial: scores normalizados por posição [1.0 → 0.0] ---
        vector_results = self._vector_search.search(query)
        vector_scores: dict[str, float] = {
            chunk.chunk_id: 1.0 - (i / len(vector_results))
            for i, chunk in enumerate(vector_results)
        }

        # --- Busca BM25: scores normalizados pelo máximo do batch [0.0 → 1.0] ---
        bm25_results = self._bm25_search.search(query, top_k=top_k)
        # Fallback para 1.0 evita divisão por zero quando todos os scores são 0
        max_bm25 = max((s for _, s in bm25_results), default=1.0) or 1.0
        bm25_scores: dict[str, float] = {
            chunk.chunk_id: score / max_bm25
            for chunk, score in bm25_results
        }

        # --- União dos resultados de ambas as buscas ---
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        # Mapa chunk_id → Chunk para reconstruir os objetos no ranking final
        chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in vector_results}
        chunk_map.update({c.chunk_id: c for c, _ in bm25_results})

        # --- Score híbrido ponderado ---
        w_sem = settings.hybrid_semantic_weight
        w_bm25 = settings.hybrid_bm25_weight

        scored = []
        for chunk_id in all_ids:
            # Chunks ausentes em uma das buscas recebem 0.0 naquela dimensão
            score = (
                w_sem * vector_scores.get(chunk_id, 0.0)
                + w_bm25 * bm25_scores.get(chunk_id, 0.0)
            )
            scored.append((chunk_map[chunk_id], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, _ in scored[:top_k]]

        logger.info(
            f"Hybrid search: {len(top_chunks)} chunks finais "
            f"(w_sem={w_sem}, w_bm25={w_bm25})"
        )
        return top_chunks
