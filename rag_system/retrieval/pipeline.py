"""Orquestrador central do pipeline RAG de recuperação e geração.

Inicializa e conecta todos os componentes do pipeline de recuperação:
    BGEEmbedder → VectorSearch + BM25Search → HybridSearch
    → BGEReranker → PromptBuilder → LLM (Groq ou Ollama)

O provedor LLM é selecionado em tempo de inicialização via llm_provider,
com suporte a fallback automático no modo 'auto'.
"""

from rag_system.ingestion.embedders.bge_embedder import BGEEmbedder
from rag_system.ingestion.vector_store.chroma_store import ChromaStore
from rag_system.retrieval.search.vector_search import VectorSearch
from rag_system.retrieval.search.bm25_search import BM25Search
from rag_system.retrieval.search.hybrid_search import HybridSearch
from rag_system.retrieval.reranker.bge_reranker import BGEReranker
from rag_system.retrieval.generator.prompt_builder import PromptBuilder
from rag_system.retrieval.generator.ollama_llm import OllamaLLM
from rag_system.retrieval.generator.groq_llm import GroqLLM
from rag_system.core.interfaces import BaseLLM
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Pipeline completo de Recuperação Aumentada por Geração (RAG).

    Orquestra o fluxo completo de uma query até a resposta final:
        1. Busca híbrida (vetorial + BM25) — top-20 candidatos
        2. Reranking com cross-encoder BGE — reduz para top-5
        3. Construção do prompt com contexto numerado
        4. Geração da resposta via LLM (Groq ou Ollama)

    Todos os componentes pesados (modelos de embedding, reranker, LLM)
    são carregados uma única vez na inicialização e reutilizados
    em todas as chamadas a query().
    """

    def __init__(self, llm_provider: str = "auto") -> None:
        """Inicializa o pipeline carregando todos os componentes.

        Args:
            llm_provider: Provedor LLM a usar. Opções:
                - ``"auto"``: tenta Ollama primeiro, cai para Groq se indisponível
                - ``"groq"``: força uso do Groq (requer GROQ_API_KEY no .env)
                - ``"ollama"``: força uso do Ollama (requer servidor local ativo)

        Raises:
            RuntimeError: Se o provedor forçado não estiver disponível,
                ou se nenhum provedor estiver acessível no modo 'auto'.
        """
        logger.info("Inicializando RAG Pipeline...")

        # --- Componentes de ingestão reutilizados na recuperação ---
        self._embedder = BGEEmbedder()
        self._store = ChromaStore()

        # --- Pipeline de busca híbrida ---
        self._vector_search = VectorSearch(self._embedder, self._store)
        self._bm25 = BM25Search()
        self._bm25.load_index()  # Carrega índice do disco — reconstruído após cada ingestão
        self._hybrid = HybridSearch(self._vector_search, self._bm25)

        # --- Reranking e geração ---
        self._reranker = BGEReranker()
        self._prompt_builder = PromptBuilder()
        self._llm: BaseLLM = self._select_llm(llm_provider)

        logger.info("RAG Pipeline pronto.")

    def _select_llm(self, provider: str) -> BaseLLM:
        """Seleciona e valida o provedor LLM conforme a estratégia configurada.

        Args:
            provider: Identificador do provedor — ``"auto"``, ``"groq"`` ou ``"ollama"``.

        Returns:
            Instância do LLM selecionado e validado como disponível.

        Raises:
            RuntimeError: Se o provedor solicitado não estiver acessível.
        """
        if provider == "groq":
            groq = GroqLLM()
            if groq.is_available():
                logger.info("LLM selecionado: Groq (forçado)")
                return groq
            raise RuntimeError("Groq solicitado mas GROQ_API_KEY não configurado no .env")

        if provider == "ollama":
            ollama_llm = OllamaLLM()
            if ollama_llm.is_available():
                logger.info("LLM selecionado: Ollama (forçado)")
                return ollama_llm
            raise RuntimeError("Ollama solicitado mas servidor não está disponível")

        # Modo auto — tenta Ollama (gratuito, local) antes de cair para Groq
        ollama_llm = OllamaLLM()
        if ollama_llm.is_available():
            logger.info("LLM selecionado: Ollama (local)")
            return ollama_llm

        groq = GroqLLM()
        if groq.is_available():
            logger.info("LLM selecionado: Groq (fallback)")
            return groq

        raise RuntimeError("Nenhum LLM disponível. Configure Ollama ou GROQ_API_KEY no .env")

    def query(self, question: str) -> dict:
        """Executa o pipeline RAG completo para uma pergunta.

        Fluxo:
            1. Valida a pergunta — rejeita queries muito curtas
            2. Busca híbrida (vetorial + BM25) → top-k candidatos
            3. Reranking com cross-encoder → top-k final
            4. Monta prompt com contexto numerado
            5. Gera resposta via LLM

        Args:
            question: Pergunta do usuário em linguagem natural.

        Returns:
            Dicionário com:
                - ``answer``: Resposta gerada pelo LLM em texto puro
                - ``sources``: Lista deduplicada de source_uris dos chunks usados
                - ``chunks``: Lista de Chunk objects do top-k final,
                  necessário para avaliação RAGAS
        """
        # Guard: queries muito curtas tendem a gerar resultados irrelevantes
        if not question or len(question.split()) < 3:
            return {
                "answer": "Por favor, faça uma pergunta mais específica.",
                "sources": [],
                "chunks": [],
            }

        logger.info(f"Query recebida: '{question}'")

        candidates = self._hybrid.search(question)
        top_chunks = self._reranker.rerank(question, candidates)
        prompt, sources = self._prompt_builder.build(question, top_chunks)
        answer = self._llm.generate(prompt)

        logger.info(f"Resposta gerada. Fontes: {sources}")
        return {"answer": answer, "sources": sources, "chunks": top_chunks}