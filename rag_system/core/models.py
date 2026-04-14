# rag_system/core/models.py

"""Modelos de dados centrais do RAG System.

Define os contratos de dados compartilhados entre os pipelines
de ingestão e recuperação.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Representa um documento bruto após parsing e limpeza.

    É o contrato de saída de todos os parsers e a entrada
    do pipeline de chunking.

    Attributes:
        doc_id: Identificador único do documento, gerado a partir
            do nome do arquivo com sufixo hash curto.
        source_uri: Caminho absoluto do arquivo de origem.
        text: Texto extraído e limpo pelo TextCleaner.
        metadata: Metadados do documento (filename, file_type,
            file_size_bytes, indexed_at, etc).
    """

    doc_id: str
    source_uri: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Representa um fragmento de documento pronto para indexação.

    É o contrato de saída dos chunkers e a unidade básica
    armazenada no ChromaDB e no índice BM25.

    Attributes:
        chunk_id: Identificador determinístico no formato
            ``{doc_id}_chunk_{index:04d}``.
        doc_id: Identificador do documento de origem — permite
            rastrear e deletar todos os chunks de um documento.
        text: Texto do fragmento.
        metadata: Metadados herdados do documento + posição do chunk
            (chunk_index, chunk_total, doc_id).
        embedding: Vetor de embedding gerado pelo BGEEmbedder.
            None até que embed_chunks() seja chamado.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None