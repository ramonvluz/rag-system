"""Chunker por parágrafos com carry-over para o RAG System.

Estratégia padrão de chunking — respeita fronteiras naturais de
parágrafos (\\n\\n) e usa carry-over para garantir continuidade
de contexto entre chunks consecutivos.
"""

from rag_system.ingestion.chunkers.base import ChunkerBase
from rag_system.core.models import Document, Chunk
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class ParagraphChunker(ChunkerBase):
    """Chunker por parágrafos com fallback por caracteres e carry-over.

    Estratégia de chunking:
        1. Divide o texto por ``\\n\\n`` (fronteiras de parágrafo)
        2. Agrupa parágrafos até atingir chunk_size
        3. Ao fechar um chunk, aplica carry-over — os últimos parágrafos
           que cabem dentro de chunk_overlap são levados para o próximo
           chunk, garantindo continuidade de contexto
        4. Parágrafos maiores que chunk_size são divididos por caracteres
           (fallback), com overlap normal entre as partes

    Os parâmetros chunk_size e chunk_overlap são lidos de settings
    e podem ser sobrescritos via .env.
    """

    def __init__(self) -> None:
        """Inicializa o chunker com parâmetros lidos de settings."""
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide um Document em Chunks respeitando fronteiras de parágrafos.

        Args:
            document: Document limpo, saída do TextCleaner.

        Returns:
            Lista de Chunks com chunk_id determinístico, carry-over
            aplicado e chunk_total preenchido em todos os metadados.
        """
        logger.info(f"Chunkizando: {document.metadata.get('filename', document.doc_id)}")

        # Divide por parágrafos — remove parágrafos vazios
        paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        index = 0
        current_parts: list[str] = []
        current_size: int = 0

        def flush() -> None:
            """Fecha o chunk atual e prepara o carry-over para o próximo.

            O carry-over percorre os parágrafos do chunk atual de trás
            para frente, acumulando os que cabem dentro de chunk_overlap.
            Isso garante que o próximo chunk comece com contexto suficiente
            para manter coerência semântica com o chunk anterior.
            """
            nonlocal index, current_parts, current_size
            if not current_parts:
                return

            text = "\n\n".join(current_parts)
            chunks.append(Chunk(
                chunk_id=self._build_chunk_id(document.doc_id, index),
                doc_id=document.doc_id,
                text=text,
                metadata=self._build_chunk_metadata(document, index, total=0),
            ))
            index += 1

            # Carry-over: percorre parágrafos de trás para frente
            # acumulando os que cabem no overlap
            overlap_parts: list[str] = []
            overlap_size = 0
            for part in reversed(current_parts):
                # +2 para o separador \n\n entre parágrafos (exceto no primeiro)
                extra = len(part) + (2 if overlap_parts else 0)
                if overlap_size + extra <= self._chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_size += extra
                else:
                    break

            current_parts = overlap_parts
            current_size = overlap_size

        for para in paragraphs:
            # Parágrafo maior que chunk_size — fallback por caracteres com overlap
            if len(para) > self._chunk_size:
                flush()
                start = 0
                while start < len(para):
                    part = para[start: start + self._chunk_size].strip()
                    if part:
                        chunks.append(Chunk(
                            chunk_id=self._build_chunk_id(document.doc_id, index),
                            doc_id=document.doc_id,
                            text=part,
                            metadata=self._build_chunk_metadata(document, index, total=0),
                        ))
                        index += 1
                    start += self._chunk_size - self._chunk_overlap
                continue

            # +2 para o separador \n\n caso já haja parágrafos acumulados
            separator = 2 if current_parts else 0
            if current_size + separator + len(para) > self._chunk_size:
                flush()

            # Recalcula separator após o flush — current_parts pode ter mudado
            separator = 2 if current_parts else 0
            current_parts.append(para)
            current_size += separator + len(para)

        # Fecha o último chunk com os parágrafos restantes
        flush()

        # Atualiza chunk_total em todos os chunks — só disponível após geração completa
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["chunk_total"] = total

        logger.info(f"{total} chunks gerados para '{document.metadata.get('filename')}'")
        return chunks