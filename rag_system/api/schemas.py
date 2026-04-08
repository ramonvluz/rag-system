"""Schemas Pydantic para validação de entrada e saída da API REST.

Define os contratos de dados das rotas /query e /ingest,
garantindo validação automática pelo FastAPI antes de qualquer
lógica de negócio ser executada.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload de entrada para a rota POST /query.

    Attributes:
        question: Pergunta do usuário em linguagem natural.
            Mínimo de 3 caracteres — validado pelo Pydantic antes
            de chegar ao pipeline RAG.
    """

    question: str = Field(..., min_length=3, description="Pergunta para o sistema RAG")


class QueryResponse(BaseModel):
    """Payload de saída da rota POST /query.

    Attributes:
        answer: Resposta gerada pelo LLM com base no contexto recuperado.
        sources: Lista deduplicada de nomes de arquivo que embasaram
            a resposta — apenas o filename, sem caminho absoluto.
    """

    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    """Payload de entrada para a rota POST /ingest.

    Attributes:
        filepath: Caminho absoluto para o arquivo a ser indexado.
            O arquivo deve existir no servidor — validado na rota.
    """

    filepath: str = Field(..., description="Caminho absoluto para o arquivo a ser indexado")


class IngestResponse(BaseModel):
    """Payload de saída da rota POST /ingest.

    Attributes:
        status: Resultado da operação — ``"success"`` ou ``"error"``.
        message: Descrição legível do resultado, incluindo contagem
            de chunks indexados em caso de sucesso.
        filepath: Caminho do arquivo processado, ecoado da requisição.
    """

    status: str
    message: str
    filepath: str