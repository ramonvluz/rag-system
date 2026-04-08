# rag_system/core/config.py
"""Configurações centralizadas do RAG System.

Todos os parâmetros são carregados via pydantic-settings, que lê
automaticamente o arquivo .env na raiz do projeto. Qualquer valor
pode ser sobrescrito por variável de ambiente sem alterar o código.

Uso:
    from rag_system.core.config import settings

    print(settings.embedding_model)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict  # ← adiciona SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Parâmetros de configuração do RAG System.

    Os caminhos de diretório são resolvidos automaticamente em
    model_post_init() com base em base_dir, mas podem ser
    sobrescritos individualmente via .env.

    Attributes:
        base_dir: Raiz do projeto, calculada a partir deste arquivo.
        data_raw_dir: Diretório dos documentos originais.
        data_processed_dir: Diretório dos documentos pós-processamento.
        chroma_db_dir: Diretório de persistência do ChromaDB.
        ingestion_log_path: Caminho do log de ingestão (.jsonl).
        bm25_index_path: Caminho do índice BM25 serializado (.pkl).
        embedding_model: Modelo de embedding (HuggingFace).
        embedding_batch_size: Número de chunks por batch no embedder.
        chunk_size: Tamanho máximo de um chunk em caracteres.
        chunk_overlap: Tamanho máximo do carry-over entre chunks.
        chroma_collection_name: Nome da coleção no ChromaDB.
        vector_search_top_k: Candidatos retornados pela busca vetorial.
        hybrid_semantic_weight: Peso da busca semântica na fusão híbrida.
        hybrid_bm25_weight: Peso da busca BM25 na fusão híbrida.
        reranker_top_k: Chunks finais retornados após reranking.
        reranker_model: Modelo cross-encoder para reranking.
        ollama_model: Modelo local do Ollama.
        ollama_base_url: URL base da API do Ollama.
        ollama_timeout: Timeout em segundos para chamadas ao Ollama.
        groq_api_key: Chave de API do Groq (obrigatória para uso remoto).
        groq_model: Modelo do Groq para geração de respostas.
        llm_provider: Provedor ativo — ``groq``, ``ollama`` ou ``auto``.
        llm_temperature: Temperatura de geração (0.0 = determinístico).
        ragas_groq_model: Modelo do Groq usado pelo avaliador RAGAS.
        ragas_max_chunks: Limite de chunks enviados ao RAGAS por amostra.
        ragas_max_chunk_chars: Limite de caracteres por chunk no RAGAS.
    """

    # ← substitui class Config — forma moderna do Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Paths ---
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_raw_dir: Optional[Path] = None
    data_processed_dir: Optional[Path] = None
    chroma_db_dir: Optional[Path] = None
    ingestion_log_path: Optional[Path] = None
    bm25_index_path: Optional[Path] = None

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16

    # --- Chunking ---
    chunk_size: int = 768
    chunk_overlap: int = 200

    # --- Vector Store ---
    chroma_collection_name: str = "rag_system"

    # --- Retrieval ---
    vector_search_top_k: int = 20
    hybrid_semantic_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    reranker_top_k: int = 5

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- LLM ---
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 60
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    llm_provider: str = "groq"
    llm_temperature: float = 0.1

    # --- Evaluation ---
    ragas_groq_model: str = "llama-3.1-8b-instant"
    ragas_max_chunks: int = 3
    ragas_max_chunk_chars: int = 500

    def model_post_init(self, __context) -> None:
        """Resolve caminhos relativos com base em base_dir.

        Executado automaticamente pelo pydantic após a inicialização.
        Só preenche os paths que não foram explicitamente definidos
        no .env, garantindo que sobrescritas parciais funcionem.
        """
        if self.data_raw_dir is None:
            self.data_raw_dir = self.base_dir / "data" / "raw"
        if self.data_processed_dir is None:
            self.data_processed_dir = self.base_dir / "data" / "processed"
        if self.chroma_db_dir is None:
            self.chroma_db_dir = self.base_dir / "chroma_db"
        if self.ingestion_log_path is None:
            self.ingestion_log_path = (
                self.base_dir / "rag_system" / "ingestion" / "ingestion_log.jsonl"
            )
        if self.bm25_index_path is None:
            self.bm25_index_path = self.base_dir / "chroma_db" / "bm25_index.pkl"


settings = Settings()