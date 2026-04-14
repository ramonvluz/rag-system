# rag_system/api/routes/ingest.py

"""Rota POST /ingest — indexação de documentos via API REST.

Expõe o pipeline de ingestão como endpoint HTTP, permitindo indexar
novos documentos sem reiniciar o servidor. Mantém singletons de
BGEEmbedder e ChromaStore para evitar recarregamento de modelos
entre requisições.
"""

import json
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rag_system.api.schemas import IngestRequest, IngestResponse
from rag_system.ingestion.parsers.factory import get_parser
from rag_system.ingestion.cleaners.text_cleaner import TextCleaner
from rag_system.ingestion.chunkers.paragraph_chunker import ParagraphChunker
from rag_system.ingestion.chunkers.table_chunker import TableChunker
from rag_system.ingestion.embedders.bge_embedder import BGEEmbedder
from rag_system.ingestion.vector_store.chroma_store import ChromaStore
from rag_system.retrieval.search.bm25_search import BM25Search
from rag_system.core.logger import get_logger
from rag_system.core.config import settings
from rag_system.core.models import Document, Chunk

logger = get_logger(__name__)

router = APIRouter()

# Singletons de módulo — inicializados na primeira requisição e reutilizados
# para evitar recarregar o modelo BGE-M3 a cada chamada ao /ingest
_embedder: BGEEmbedder | None = None
_store: ChromaStore | None = None


def get_components() -> tuple[BGEEmbedder, ChromaStore]:
    """Retorna os componentes pesados, inicializando-os na primeira chamada.

    Returns:
        Tupla (embedder, store) prontos para uso.
    """
    global _embedder, _store
    if _embedder is None:
        _embedder = BGEEmbedder()
    if _store is None:
        _store = ChromaStore()
    return _embedder, _store


def _save_processed_document(doc: Document) -> Path:
    """Persiste o Document limpo em data/processed/ como JSON.

    Args:
        doc: Document limpo, saída do TextCleaner.

    Returns:
        Path do arquivo JSON gerado.
    """
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.data_processed_dir / f"{doc.doc_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(doc), f, ensure_ascii=False, indent=2)
    logger.info(f"Documento processado salvo em '{output_path}'")
    return output_path


def _rebuild_bm25(store: ChromaStore) -> None:
    """Reconstrói o índice BM25 após ingestão via API.

    Necessário para manter o BM25 sincronizado com o ChromaDB após
    cada novo documento indexado. Inclui guard contra corpus vazio.

    Args:
        store: Instância do ChromaStore com os chunks já atualizados.
    """
    logger.info("Reconstruindo índice BM25 após ingestão via API...")
    results = store._collection.get(include=["documents", "metadatas"])

    if not results["ids"]:
        # Guard: BM25Okapi lança ZeroDivisionError com corpus vazio
        logger.warning("Nenhum chunk no ChromaDB — índice BM25 não reconstruído.")
        return

    chunks = [
        Chunk(
            chunk_id=cid,
            doc_id=meta.get("doc_id", ""),
            text=text,
            metadata=meta,
        )
        for cid, text, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        )
    ]
    bm25 = BM25Search()
    bm25.build_index(chunks)
    logger.info(f"✅ Índice BM25 reconstruído com {len(chunks)} chunks.")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Indexa um documento no ChromaDB e reconstrói o índice BM25.

    Executa o pipeline completo: parse → clean → persist → chunk →
    embed → upsert → rebuild BM25 → log.

    Args:
        request: Payload com o caminho absoluto do arquivo.

    Returns:
        IngestResponse com status, contagem de chunks e filepath.

    Raises:
        HTTPException 404: Se o arquivo não existir no caminho informado.
        HTTPException 500: Se ocorrer qualquer erro durante a ingestão.
    """
    logger.info(f"POST /ingest — '{request.filepath}'")
    filepath = Path(request.filepath)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {request.filepath}")

    try:
        embedder, store = get_components()
        cleaner = TextCleaner()

        # Seleciona chunker pelo tipo de arquivo
        ext = filepath.suffix.lower()
        chunker = TableChunker() if ext in [".csv", ".xlsx"] else ParagraphChunker()

        parser = get_parser(str(filepath))
        doc = parser.parse(str(filepath))
        doc = cleaner.clean(doc)
        _save_processed_document(doc)
        chunks = chunker.chunk(doc)
        chunks = embedder.embed_chunks(chunks)
        store.upsert(chunks)
        _rebuild_bm25(store)

        # Registra resultado no log de ingestão
        log_entry = {
            "doc_id": filepath.stem,
            "source_uri": str(filepath.resolve()),
            "chunks_gerados": len(chunks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
        }
        settings.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.ingestion_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return IngestResponse(
            status="success",
            message=f"{len(chunks)} chunks indexados com sucesso.",
            filepath=str(filepath),
        )
    except Exception as e:
        logger.error(f"Erro no /ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))