# rag_system/ingestion/chunkers/table_chunker.py

"""Chunker especializado para tabelas CSV e XLSX do RAG System.

Processa documentos tabulares em formato Markdown (saída de CSVParser
e XLSXParser), gerando um chunk por linha de dados com o cabeçalho
embutido — permitindo que cada chunk seja semanticamente autossuficiente
para busca e recuperação.
"""

from rag_system.ingestion.chunkers.base import ChunkerBase
from rag_system.core.models import Document, Chunk
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class TableChunker(ChunkerBase):
    """Chunker para tabelas — cada linha de dados vira um chunk independente.

    Estratégia de chunking:
        1. Extrai o cabeçalho (primeira linha não vazia da tabela Markdown)
        2. Ignora linhas separadoras do Markdown (``|---|---|``)
        3. Para cada linha de dados, cria um chunk com o formato::

            {cabeçalho}
            {linha de dados}

    Embutir o cabeçalho em cada chunk garante que o modelo de embedding
    e o LLM entendam o significado de cada coluna sem depender de
    contexto externo — fundamental para tabelas com muitas colunas.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide um Document tabular em Chunks — um por linha de dados.

        Espera que document.text esteja em formato Markdown tabular,
        conforme gerado por CSVParser e XLSXParser (pandas to_markdown).

        Args:
            document: Document com texto em Markdown tabular.

        Returns:
            Lista de Chunks onde cada item contém o cabeçalho da tabela
            e uma linha de dados, com chunk_total igual ao número de
            linhas de dados (excluindo cabeçalho e separadores).
        """
        logger.info(f"Chunkizando tabela: {document.metadata.get('filename', document.doc_id)}")
        lines = document.text.split("\n")

        # Percorre as linhas para separar cabeçalho de dados
        # ignorando linhas vazias e separadores Markdown (|---|---|)
        header = ""
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("|---"):
                continue
            if not header:
                # Primeira linha não vazia é o cabeçalho da tabela
                header = stripped
            else:
                data_lines.append(stripped)

        chunks = []
        for index, line in enumerate(data_lines):
            if not line:
                continue

            # Cabeçalho embutido garante que o chunk seja autossuficiente
            chunk_text = f"{header}\n{line}"
            chunks.append(Chunk(
                chunk_id=self._build_chunk_id(document.doc_id, index),
                doc_id=document.doc_id,
                text=chunk_text,
                metadata=self._build_chunk_metadata(document, index, total=len(data_lines)),
            ))

        logger.info(f"{len(chunks)} chunks de tabela gerados para '{document.metadata.get('filename')}'")
        return chunks