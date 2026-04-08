"""Busca lexical usando o algoritmo BM25 (Best Match 25).

Implementa a etapa esparsa do pipeline de busca híbrida — eficaz para
termos exatos como siglas, nomes próprios e palavras-chave específicas
que a busca semântica por similaridade de cosseno tende a perder.

O índice é construído após cada ingestão e persistido em disco como
arquivo pickle, evitando reconstrução a cada inicialização da API.
"""

import pickle

from rank_bm25 import BM25Okapi

from rag_system.core.models import Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class BM25Search:
    """Busca lexical BM25 com persistência em disco.

    Complementa o VectorSearch no pipeline híbrido — enquanto a busca
    vetorial captura similaridade semântica, o BM25 captura correspondência
    exata de termos, tornando a busca híbrida mais robusta.

    O índice é mantido em memória após carregamento e persistido em disco
    em formato pickle (bm25_index.pkl) para reutilização entre sessões.
    """

    def __init__(self) -> None:
        """Inicializa BM25Search sem índice carregado.

        O índice deve ser populado via build_index() ou load_index()
        antes de qualquer chamada a search().
        """
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """Constrói o índice BM25 a partir dos chunks e persiste em disco.

        Tokeniza o texto de cada chunk por espaços (lowercase) para
        compatibilidade com BM25Okapi. O índice e os chunks são serializados
        juntos em pickle para permitir reconstrução exata dos objetos Chunk
        no load_index().

        Args:
            chunks: Lista completa de Chunks indexados no ChromaDB.
                Deve conter pelo menos um chunk — BM25Okapi lança
                ZeroDivisionError com corpus vazio.
        """
        logger.info(f"Construindo índice BM25 com {len(chunks)} chunks...")
        self._chunks = chunks

        # Tokenização simples por espaços em lowercase — consistente com search()
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

        settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            # Persiste bm25 e chunks juntos — necessário para reconstruir Chunk objects
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

        logger.info(f"Índice BM25 persistido em '{settings.bm25_index_path}'")

    def load_index(self) -> bool:
        """Carrega o índice BM25 do disco para a memória.

        Chamado na inicialização da API para evitar reconstrução do índice
        a cada restart. Se o arquivo não existir (ex: após make reset),
        retorna False sem lançar exceção — o índice será reconstruído
        na próxima ingestão.

        Returns:
            True se o índice foi carregado com sucesso,
            False se o arquivo não existir em disco.
        """
        if not settings.bm25_index_path.exists():
            logger.warning("Índice BM25 não encontrado em disco.")
            return False

        with open(settings.bm25_index_path, "rb") as f:
            data = pickle.load(f)

        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        logger.info(f"Índice BM25 carregado: {len(self._chunks)} chunks.")
        return True

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Executa busca BM25 e retorna chunks com seus scores de relevância.

        A tokenização da query usa o mesmo critério do build_index()
        (lowercase + split por espaços) para garantir compatibilidade.

        Args:
            query: Pergunta ou texto de busca do usuário.
            top_k: Número máximo de resultados a retornar.

        Returns:
            Lista de tuplas (Chunk, score) ordenada por relevância
            BM25 decrescente. Retorna lista vazia se o índice não
            estiver carregado.
        """
        if self._bm25 is None:
            logger.error("Índice BM25 não carregado. Chame load_index() ou build_index() primeiro.")
            return []

        # Tokenização idêntica à usada no build_index — lowercase + split
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Ordena índices por score decrescente e seleciona top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self._chunks[i], float(scores[i])) for i in top_indices]

        logger.info(f"{len(results)} chunks recuperados via BM25.")
        return results