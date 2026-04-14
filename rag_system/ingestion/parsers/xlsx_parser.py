# rag_system/ingestion/parsers/xlsx_parser.py

"""Parser de arquivos Excel (.xlsx) usando pandas.

Lê todas as abas (sheets) do arquivo e converte cada uma em formato
Markdown tabular, separadas por cabeçalho de seção. Compatível com
o TableChunker que processa cada linha como chunk independente.
"""

import pandas as pd

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class XLSXParser(ParserBase):
    """Parser de arquivos Excel (.xlsx) via pandas.

    Processa todas as abas do arquivo, convertendo cada uma em
    Markdown tabular com cabeçalho de seção (## Sheet: nome).
    Arquivos com múltiplas abas geram um Document único com
    todas as tabelas concatenadas.
    """

    def parse(self, filepath: str) -> Document:
        """Lê todas as abas de um arquivo XLSX e retorna um Document.

        Cada aba é convertida em Markdown tabular precedido por um
        cabeçalho ``## Sheet: {nome}``, permitindo ao TableChunker
        identificar a origem de cada linha durante o chunking.

        Valores ausentes (NaN) são substituídos por string vazia para
        evitar erros de serialização downstream no pipeline.

        Args:
            filepath: Caminho absoluto para o arquivo .xlsx.

        Returns:
            Document com text contendo todas as abas em Markdown e
            metadata incluindo a lista de nomes das abas (sheets).
        """
        logger.info(f"Parsing XLSX: {filepath}")
        xl = pd.ExcelFile(filepath)
        parts = []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet).fillna("")
            # Cada aba recebe um cabeçalho de seção para rastreabilidade no chunking
            parts.append(f"## Sheet: {sheet}\n{df.to_markdown(index=False)}")

        # Abas separadas por linha em branco para facilitar leitura pelo chunker
        text = "\n\n".join(parts)
        doc_id = self._generate_doc_id(filepath)
        metadata = self._build_metadata(
            filepath,
            file_type="xlsx",
            sheets=xl.sheet_names,
        )

        return Document(doc_id=doc_id, source_uri=filepath, text=text, metadata=metadata)
