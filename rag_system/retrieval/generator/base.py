# rag_system/retrieval/generator/base.py

"""Classe base concreta para todos os LLMs do RAG System.

Segue o mesmo padrão de ParserBase e ChunkerBase — camada intermediária
entre a ABC (BaseLLM) e as implementações concretas (GroqLLM, OllamaLLM),
permitindo adicionar helpers compartilhados sem modificar a interface.
"""

from rag_system.core.interfaces import BaseLLM
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class LLMBase(BaseLLM):
    """Classe base concreta para LLMs — pronta para helpers compartilhados.

    Atualmente serve como camada de indireção entre BaseLLM e as
    implementações concretas. Helpers comuns a múltiplos LLMs devem
    ser adicionados aqui — ex: sanitização de prompt, truncamento
    por limite de tokens, retry com backoff exponencial.
    """

    def generate(self, prompt: str) -> str:
        """Gera uma resposta em texto a partir de um prompt.

        Args:
            prompt: Prompt montado pelo PromptBuilder.

        Returns:
            Resposta gerada pelo modelo em texto puro.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar generate()")

    def is_available(self) -> bool:
        """Verifica se o modelo está acessível para lógica de fallback.

        Returns:
            True se o modelo pode receber requisições, False caso contrário.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar is_available()")