# rag_system/ingestion/parsers/docx_parser.py

"""Parser de documentos DOCX usando Docling.

Utiliza o Docling (DocumentConverter) para extrair texto de arquivos
Word (.docx), exportando o conteúdo em Markdown para preservar
estrutura de títulos, parágrafos e tabelas.
"""

from docling.document_converter import DocumentConverter

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class DOCXParser(ParserBase):
    """Parser de arquivos Word (.docx) via Docling.

    Converte documentos Word em Markdown estruturado, preservando
    títulos, parágrafos, listas e tabelas presentes no arquivo original.
    """

    def __init__(self) -> None:
        """Inicializa o conversor Docling com configurações padrão."""
        self._converter = DocumentConverter()

    def parse(self, filepath: str) -> Document:
        """Extrai texto de um arquivo DOCX e retorna um Document.

        O texto é exportado em Markdown pelo Docling para preservar
        a estrutura do documento, melhorando a qualidade do chunking
        posterior.

        Args:
            filepath: Caminho absoluto para o arquivo .docx.

        Returns:
            Document com text em Markdown e metadata preenchida.
        """
        logger.info(f"Parsing DOCX: {filepath}")
        result = self._converter.convert(filepath)

        # Exporta como Markdown para preservar estrutura do documento
        text = result.document.export_to_markdown()
        doc_id = self._generate_doc_id(filepath)
        metadata = self._build_metadata(filepath, file_type="docx")

        return Document(doc_id=doc_id, source_uri=filepath, text=text, metadata=metadata)
