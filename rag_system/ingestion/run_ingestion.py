"""Pipeline de ingestão do RAG System — entrada pelo CLI.

Orquestra as etapas de parsing, limpeza, persistência em
data/processed/, chunking, embedding e indexação no ChromaDB e BM25.

Uso via Makefile:
    make ingest FILE=data/raw/documento.pdf   # arquivo único
    make ingest-all                            # todos os arquivos em data/raw/

Uso direto:
    python -m rag_system.ingestion.run_ingestion --file data/raw/doc.pdf
    python -m rag_system.ingestion.run_ingestion --all
"""

import argparse
import json
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

from rag_system.ingestion.parsers.factory import get_parser, PARSER_MAP
from rag_system.ingestion.cleaners.text_cleaner import TextCleaner
from rag_system.ingestion.chunkers.paragraph_chunker import ParagraphChunker
from rag_system.ingestion.chunkers.table_chunker import TableChunker
from rag_system.ingestion.embedders.bge_embedder import BGEEmbedder
from rag_system.ingestion.vector_store.chroma_store import ChromaStore
from rag_system.retrieval.search.bm25_search import BM25Search
from rag_system.core.config import settings
from rag_system.core.logger import get_logger
from rag_system.core.models import Document, Chunk

logger = get_logger(__name__)


def get_chunker(filepath: str) -> ParagraphChunker | TableChunker:
    """Seleciona o chunker adequado com base na extensão do arquivo.

    Arquivos tabulares (CSV, XLSX) usam TableChunker — um chunk por linha.
    Todos os outros formatos usam ParagraphChunker — chunks por parágrafos.

    Args:
        filepath: Caminho do arquivo a ser chunkizado.

    Returns:
        Instância de TableChunker para .csv/.xlsx,
        ou ParagraphChunker para os demais formatos.
    """
    ext = Path(filepath).suffix.lower()
    if ext in [".csv", ".xlsx"]:
        return TableChunker()
    return ParagraphChunker()


def _save_processed_document(doc: Document) -> Path:
    """Persiste o Document limpo em data/processed/ como JSON.

    Cria o diretório se não existir. A operação é idempotente —
    re-ingerir o mesmo arquivo sobrescreve o JSON anterior.

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


def ingest_file(
    filepath: str,
    cleaner: TextCleaner,
    embedder: BGEEmbedder,
    store: ChromaStore,
) -> str:
    """Executa o pipeline completo de ingestão para um único arquivo.

    Pipeline:
        Parser → TextCleaner → _save_processed_document →
        Chunker → BGEEmbedder → ChromaStore

    Erros são capturados e logados sem interromper a ingestão dos
    demais arquivos em modo ``--all``. O status é sempre registrado
    no ingestion_log.jsonl independentemente do resultado.

    Args:
        filepath: Caminho absoluto do arquivo a ser indexado.
        cleaner: Instância compartilhada do TextCleaner.
        embedder: Instância compartilhada do BGEEmbedder.
        store: Instância compartilhada do ChromaStore.

    Returns:
        ``"success"`` se a ingestão foi concluída sem erros,
        ``"error"`` caso contrário.
    """
    logger.info(f"Iniciando ingestão: {filepath}")
    status = "success"
    chunks_count = 0

    try:
        parser = get_parser(filepath)
        doc = parser.parse(filepath)
        doc = cleaner.clean(doc)
        _save_processed_document(doc)
        chunker = get_chunker(filepath)
        chunks = chunker.chunk(doc)
        chunks = embedder.embed_chunks(chunks)
        store.upsert(chunks)
        chunks_count = len(chunks)
        logger.info(f"✅ Ingestão concluída: {chunks_count} chunks indexados.")
    except Exception as e:
        logger.error(f"❌ Erro na ingestão de '{filepath}': {e}")
        status = "error"

    # Registra resultado no log de ingestão — sempre, mesmo em caso de erro
    log_entry = {
        "doc_id": Path(filepath).stem,
        "source_uri": str(Path(filepath).resolve()),
        "chunks_gerados": chunks_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    settings.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.ingestion_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return status


def rebuild_bm25(store: ChromaStore) -> None:
    """Reconstrói o índice BM25 a partir de todos os chunks no ChromaDB.

    Chamado após cada ingestão — garante que o índice BM25 esteja
    sempre sincronizado com o estado atual do ChromaDB.
    Não faz nada se o ChromaDB estiver vazio, evitando ZeroDivisionError
    no BM25Okapi com corpus vazio.

    Args:
        store: Instância do ChromaStore com chunks já indexados.
    """
    logger.info("Reconstruindo índice BM25...")
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


def main() -> None:
    """Ponto de entrada do CLI de ingestão.

    Inicializa os componentes compartilhados uma única vez e os
    reutiliza para todos os arquivos — evita recarregar o modelo
    de embedding (BGE-M3) a cada arquivo em modo ``--all``.
    """
    parser = argparse.ArgumentParser(description="RAG-System — Pipeline de Ingestão")
    parser.add_argument("--file", type=str, help="Caminho para um arquivo específico")
    parser.add_argument("--all", action="store_true", help="Indexa todos os arquivos em data/raw/")
    args = parser.parse_args()

    # Componentes pesados inicializados uma única vez e compartilhados entre arquivos
    cleaner = TextCleaner()
    embedder = BGEEmbedder()
    store = ChromaStore()

    if args.file:
        filepath = str(Path(args.file).resolve())
        ingest_file(filepath, cleaner, embedder, store)
        rebuild_bm25(store)

    elif args.all:
        supported_exts = set(PARSER_MAP.keys())
        files = [f for f in settings.data_raw_dir.iterdir() if f.suffix.lower() in supported_exts]

        if not files:
            logger.warning(f"Nenhum arquivo suportado encontrado em '{settings.data_raw_dir}'")
            return

        logger.info(f"{len(files)} arquivo(s) encontrado(s) para indexar.")
        for f in files:
            ingest_file(str(f), cleaner, embedder, store)

        # BM25 reconstruído uma única vez após todos os arquivos — mais eficiente
        rebuild_bm25(store)
        logger.info(f"Total no ChromaDB: {store.count} chunks")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()