"""Interfaces abstratas (ABCs) do RAG System.

Define os contratos que todas as implementações concretas devem seguir,
permitindo trocar qualquer componente — ex: ChromaDB → Qdrant,
Groq → OpenAI — sem alterar o restante do sistema.
"""

from abc import ABC, abstractmethod
from rag_system.core.models import Document, Chunk


class BaseParser(ABC):
    """Interface para parsers de documentos.

    Toda implementação concreta deve converter um arquivo em um
    Document padronizado, independente do formato de origem.
    """

    @abstractmethod
    def parse(self, filepath: str) -> Document:
        """Lê e converte um arquivo em Document.

        Args:
            filepath: Caminho absoluto para o arquivo de origem.

        Returns:
            Document com text extraído e metadata preenchida.
        """


class BaseChunker(ABC):
    """Interface para chunkers de documentos.

    Toda implementação concreta deve dividir um Document limpo
    em uma lista de Chunks prontos para embedding.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Divide um Document em Chunks.

        Args:
            document: Document limpo, saída do TextCleaner.

        Returns:
            Lista de Chunks com chunk_id determinístico e metadados.
        """


class BaseVectorStore(ABC):
    """Interface para bancos de dados vetoriais.

    Toda implementação concreta deve suportar upsert idempotente,
    busca por similaridade e deleção por doc_id.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insere ou atualiza chunks no banco vetorial.

        A operação é idempotente — re-indexar o mesmo documento
        não cria duplicatas, pois usa chunk_id como chave.

        Args:
            chunks: Lista de Chunks com embedding preenchido.
        """

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """Busca os chunks mais similares ao embedding da query.

        Args:
            query_embedding: Vetor de embedding da query.
            top_k: Número máximo de chunks a retornar.

        Returns:
            Lista de até top_k Chunks ordenados por similaridade.
        """

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove todos os chunks de um documento.

        Args:
            doc_id: Identificador do documento cujos chunks
                serão removidos.
        """


class BaseLLM(ABC):
    """Interface para modelos de linguagem (LLMs).

    Toda implementação concreta deve suportar geração de texto
    e verificação de disponibilidade para lógica de fallback.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Gera uma resposta em texto a partir de um prompt.

        Args:
            prompt: Prompt montado pelo PromptBuilder.

        Returns:
            Resposta gerada pelo modelo em texto puro.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica se o modelo está acessível.

        Usado pela lógica de fallback do pipeline — se Ollama
        não estiver disponível, o sistema cai para Groq.

        Returns:
            True se o modelo pode receber requisições, False caso contrário.
        """