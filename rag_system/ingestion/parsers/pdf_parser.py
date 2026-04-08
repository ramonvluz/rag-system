"""Parser de documentos PDF usando Docling.

Utiliza o Docling (DocumentConverter) para extrair texto de PDFs,
incluindo suporte a OCR para documentos digitalizados, e exporta
o conteúdo em Markdown para preservar estrutura de títulos e tabelas.
"""

from docling.document_converter import DocumentConverter

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class PDFParser(ParserBase):
    """Parser de arquivos PDF via Docling.

    Converte PDFs em Markdown estruturado, preservando títulos,
    parágrafos e tabelas. Suporta PDFs digitais e digitalizados
    (via OCR com RapidOCR integrado ao Docling).
    """

    def __init__(self) -> None:
        """Inicializa o conversor Docling com configurações padrão."""
        self._converter = DocumentConverter()

    def parse(self, filepath: str) -> Document:
        """Extrai texto de um arquivo PDF e retorna um Document.

        O texto é exportado em Markdown pelo Docling para preservar
        a estrutura do documento (títulos, listas, tabelas), que
        melhora a qualidade do chunking posterior.

        Args:
            filepath: Caminho absoluto para o arquivo .pdf.

        Returns:
            Document com text em Markdown e metadata preenchida.
        """
        logger.info(f"Parsing PDF: {filepath}")
        result = self._converter.convert(filepath)

        # Exporta como Markdown para preservar estrutura do documento
        text = result.document.export_to_markdown()
        doc_id = self._generate_doc_id(filepath)
        metadata = self._build_metadata(filepath, file_type="pdf")

        return Document(doc_id=doc_id, source_uri=filepath, text=text, metadata=metadata)
