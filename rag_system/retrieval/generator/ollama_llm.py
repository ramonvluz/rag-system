# rag_system/retrieval/generator/ollama_llm.py

"""Implementação do LLM usando Ollama local.

Usa o cliente ollama para gerar respostas via modelos rodando
localmente (ex: llama3.2:3b). Não requer internet nem API key —
apenas o servidor Ollama em execução em ollama_base_url.
"""

import ollama

from rag_system.retrieval.generator.base import LLMBase
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(LLMBase):
    """LLM local via Ollama.

    Usado como provedor preferencial em llm_provider=auto — gratuito,
    sem latência de rede e sem dependência de API key. Cai para Groq
    se o servidor Ollama não estiver acessível.
    """

    def is_available(self) -> bool:
        """Verifica disponibilidade fazendo uma chamada real ao servidor Ollama.

        Tenta listar os modelos disponíveis via ollama.list() — se o
        servidor não estiver rodando, a chamada lança uma exceção que
        é capturada e retorna False.

        Returns:
            True se o servidor Ollama estiver acessível, False caso contrário.
        """
        try:
            ollama.list()
            return True
        except Exception:
            # Servidor não está rodando ou inacessível — fallback para Groq
            return False

    def generate(self, prompt: str) -> str:
        """Gera resposta via Ollama com o modelo local configurado.

        Args:
            prompt: Prompt montado pelo PromptBuilder.

        Returns:
            Resposta gerada pelo modelo em texto puro.

        Raises:
            Exception: Propaga exceções do Ollama após logar o erro.
        """
        logger.info(f"Gerando resposta com Ollama ({settings.ollama_model})...")
        try:
            response = ollama.generate(
                model=settings.ollama_model,
                prompt=prompt,
                options={"temperature": settings.llm_temperature},
            )
            return response["response"].strip()
        except Exception as e:
            logger.error(f"Erro no Ollama: {e}")
            raise