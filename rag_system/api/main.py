# rag_system/api/main.py

"""Ponto de entrada da API REST do RAG System.

Inicializa a aplicação FastAPI, registra os routers de query e ingest,
e expõe o endpoint de health check.

Execução:
    uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload
    ou: make run
"""

from fastapi import FastAPI

from rag_system.api.routes.query import router as query_router
from rag_system.api.routes.ingest import router as ingest_router
from rag_system.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Sistema RAG com busca híbrida, BGE-M3 e Llama 3.2 local.",
    version="0.1.0",
)

# Registra routers — cada um gerencia seus próprios componentes singleton
app.include_router(query_router, tags=["Query"])
app.include_router(ingest_router, tags=["Ingest"])


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Verifica se a API está no ar.

    Returns:
        Dicionário com status ``"ok"`` e versão atual da API.
    """
    return {"status": "ok", "version": "0.1.0"}