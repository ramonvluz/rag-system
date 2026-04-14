# rag_system/retrieval/generator/groq_llm.py

"""Implementação do LLM usando a API remota do Groq.

Usa o cliente oficial da Groq para gerar respostas via modelos
hospedados na infraestrutura Groq (ex: llama-3.1-8b-instant).
Requer GROQ_API_KEY configurada no .env.
"""

from groq import Groq

from rag_system.retrieval.generator.base import LLMBase
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class GroqLLM(LLMBase):
    """LLM remoto via API Groq.

    Usado como provedor padrão (llm_provider=groq) ou como fallback
    quando o Ollama local não está disponível (llm_provider=auto).
    Requer conexão com a internet e GROQ_API_KEY válida.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Groq com a API key de settings."""
        self._client = Groq(api_key=settings.groq_api_key)

    def is_available(self) -> bool:
        """Verifica disponibilidade checando se a API key está configurada.

        Não faz chamada de rede — apenas valida se a chave existe.
        Uma chave inválida só será detectada na primeira chamada a generate().

        Returns:
            True se GROQ_API_KEY estiver definida e não vazia.
        """
        return bool(settings.groq_api_key)

    def generate(self, prompt: str) -> str:
        """Gera resposta via API Groq com o modelo configurado.

        Args:
            prompt: Prompt montado pelo PromptBuilder.

        Returns:
            Resposta gerada pelo modelo em texto puro.

        Raises:
            Exception: Propaga exceções da API Groq após logar o erro
                — permite que o pipeline trate o fallback para Ollama.
        """
        logger.info(f"Gerando resposta com Groq ({settings.groq_model})...")
        try:
            response = self._client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erro no Groq: {e}")
            raise