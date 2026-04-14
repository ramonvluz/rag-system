# rag_system/api/routes/query.py

"""Rota POST /query — recuperação e geração de respostas via API REST.

Expõe o RAGPipeline como endpoint HTTP. O pipeline é inicializado
uma única vez no carregamento do módulo e reutilizado em todas as
requisições — evita recarregar modelos pesados entre chamadas.
"""

from fastapi import APIRouter, HTTPException

from rag_system.api.schemas import QueryRequest, QueryResponse
from rag_system.retrieval.pipeline import RAGPipeline
from rag_system.core.logger import get_logger
from rag_system.core.config import settings

logger = get_logger(__name__)

router = APIRouter()

# Singleton inicializado no carregamento do módulo — modelos BGE-M3,
# BGEReranker e LLM são carregados uma única vez e compartilhados
_pipeline: RAGPipeline = RAGPipeline(llm_provider=settings.llm_provider)


def get_pipeline() -> RAGPipeline:
    """Retorna o pipeline singleton, inicializando-o se necessário.

    O guard ``if _pipeline is None`` cobre o caso de recriação
    explícita em testes — em produção o pipeline já está inicializado.

    Returns:
        Instância ativa do RAGPipeline.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(llm_provider=settings.llm_provider)
    return _pipeline


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Executa o pipeline RAG e retorna a resposta gerada pelo LLM.

    Args:
        request: Payload com a pergunta do usuário (mín. 3 caracteres).

    Returns:
        QueryResponse com a resposta gerada e as fontes consultadas
        — apenas o nome do arquivo, sem caminho absoluto.

    Raises:
        HTTPException 500: Se ocorrer qualquer erro no pipeline RAG.
    """
    logger.info(f"POST /query — '{request.question}'")
    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.question)

        # Expõe apenas o filename — evita vazar caminhos absolutos do servidor
        sources = [s.split("/")[-1] for s in result["sources"]]

        return QueryResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        logger.error(f"Erro no /query: {e}")
        raise HTTPException(status_code=500, detail=str(e))