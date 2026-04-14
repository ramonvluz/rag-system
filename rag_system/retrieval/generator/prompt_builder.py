# rag_system/retrieval/generator/prompt_builder.py

"""Construtor de prompts para o pipeline RAG.

Monta o prompt final injetando os chunks recuperados como contexto
e retorna as fontes para auditabilidade da resposta.
"""

from rag_system.core.models import Chunk
from rag_system.core.logger import get_logger

logger = get_logger(__name__)

# Instrui o modelo a responder apenas com base no contexto fornecido,
# evitando alucinações e garantindo rastreabilidade das respostas
PROMPT_TEMPLATE = """Você é um assistente especializado. Responda à pergunta do usuário com base APENAS no contexto fornecido abaixo.
Se a informação não estiver no contexto, diga claramente que não encontrou a informação.
Não invente informações. Responda em português.

---
CONTEXTO:
{context}
---

PERGUNTA: {query}

RESPOSTA:"""


class PromptBuilder:
    """Monta prompts RAG injetando contexto recuperado e rastreando fontes.

    O template instrui explicitamente o modelo a não inventar informações
    e a responder apenas com base no contexto — estratégia fundamental
    para reduzir alucinações em sistemas RAG.
    """

    def build(self, query: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
        """Monta o prompt final com contexto numerado e coleta as fontes.

        Cada chunk é numerado [1], [2], ... no contexto para facilitar
        eventual citação pelo modelo. Fontes duplicadas são deduplicadas
        preservando a ordem de aparição.

        Args:
            query: Pergunta original do usuário.
            chunks: Top-k chunks selecionados pelo BGEReranker.

        Returns:
            Tupla (prompt, sources) onde:
                - prompt: String pronta para envio ao LLM
                - sources: Lista deduplicada de source_uris dos chunks,
                  usada para preencher o campo sources da QueryResponse
        """
        context_parts = []
        sources = []

        for i, chunk in enumerate(chunks):
            # Numera cada chunk para referência no contexto
            context_parts.append(f"[{i + 1}] {chunk.text}")
            source = chunk.metadata.get("source_uri", chunk.doc_id)
            # Deduplica fontes preservando ordem de aparição
            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)

        logger.debug(f"Prompt montado: {len(prompt)} chars | {len(sources)} fonte(s)")
        return prompt, sources
