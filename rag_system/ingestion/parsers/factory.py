"""Fábrica de parsers do RAG System (Factory Pattern).

Centraliza o mapeamento entre extensões de arquivo e parsers concretos.
Adicionar suporte a um novo formato requer apenas criar o parser
e registrá-lo em PARSER_MAP — nenhum outro arquivo precisa mudar.
"""

from pathlib import Path

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.ingestion.parsers.pdf_parser import PDFParser
from rag_system.ingestion.parsers.docx_parser import DOCXParser
from rag_system.ingestion.parsers.xlsx_parser import XLSXParser
from rag_system.ingestion.parsers.csv_parser import CSVParser
from rag_system.ingestion.parsers.html_parser import HTMLParser
from rag_system.core.logger import get_logger

logger = get_logger(__name__)

# Mapeamento extensão → classe de parser
# Para adicionar um novo formato: crie o parser e registre aqui
PARSER_MAP: dict[str, type[ParserBase]] = {
    ".pdf":  PDFParser,
    ".docx": DOCXParser,
    ".xlsx": XLSXParser,
    ".csv":  CSVParser,
    ".html": HTMLParser,
    ".htm":  HTMLParser,  # Alias para .html — mesmo parser
}


def get_parser(filepath: str) -> ParserBase:
    """Retorna o parser adequado para o formato do arquivo.

    Seleciona o parser com base na extensão do arquivo e retorna
    uma instância pronta para uso. A seleção é case-insensitive
    — ``.PDF`` e ``.pdf`` resolvem para o mesmo parser.

    Args:
        filepath: Caminho para o arquivo a ser parseado.

    Returns:
        Instância do parser correspondente à extensão do arquivo.

    Raises:
        ValueError: Se a extensão não estiver registrada em PARSER_MAP.

    Example:
        >>> parser = get_parser("data/raw/relatorio.pdf")
        >>> doc = parser.parse("data/raw/relatorio.pdf")
    """
    ext = Path(filepath).suffix.lower()
    parser_class = PARSER_MAP.get(ext)

    if not parser_class:
        raise ValueError(
            f"Formato não suportado: '{ext}'. Suportados: {list(PARSER_MAP.keys())}"
        )

    logger.debug(f"Parser selecionado para '{ext}': {parser_class.__name__}")
    return parser_class()