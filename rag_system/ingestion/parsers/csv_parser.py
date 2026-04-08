"""Parser de arquivos CSV usando pandas.

Lê arquivos CSV com pandas e converte o conteúdo em formato Markdown
tabular, compatível com o TableChunker que processa cada linha
como um chunk independente com cabeçalho embutido.
"""

import pandas as pd

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class CSVParser(ParserBase):
    """Parser de arquivos CSV via pandas.

    Converte tabelas CSV em Markdown estruturado. O formato Markdown
    é consumido pelo TableChunker, que extrai o cabeçalho e gera
    um chunk por linha de dados com o cabeçalho embutido.
    """

    def parse(self, filepath: str) -> Document:
        """Lê um arquivo CSV e retorna um Document com tabela em Markdown.

        Valores ausentes (NaN) são substituídos por string vazia para
        evitar erros de serialização downstream no pipeline.

        Args:
            filepath: Caminho absoluto para o arquivo .csv.

        Returns:
            Document com text em Markdown tabular e metadata contendo
            o número de linhas e a lista de colunas do arquivo.
        """
        logger.info(f"Parsing CSV: {filepath}")
        df = pd.read_csv(filepath).fillna("")

        # Converte para Markdown tabular — formato esperado pelo TableChunker
        text = df.to_markdown(index=False)
        doc_id = self._generate_doc_id(filepath)
        metadata = self._build_metadata(
            filepath,
            file_type="csv",
            rows=len(df),
            columns=list(df.columns),
        )

        return Document(doc_id=doc_id, source_uri=filepath, text=text, metadata=metadata)
