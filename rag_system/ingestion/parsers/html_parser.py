"""Parser de arquivos HTML usando BeautifulSoup.

Extrai o conteúdo textual de páginas HTML, removendo elementos
de navegação e estrutura (scripts, estilos, nav, header, footer)
para preservar apenas o conteúdo semântico relevante.
"""

from pathlib import Path

from bs4 import BeautifulSoup

from rag_system.ingestion.parsers.base import ParserBase
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class HTMLParser(ParserBase):
    """Parser de arquivos HTML via BeautifulSoup.

    Remove tags de estrutura e navegação antes da extração de texto,
    garantindo que apenas o conteúdo semântico da página seja
    indexado — sem poluição de menus, rodapés ou scripts inline.
    """

    def parse(self, filepath: str) -> Document:
        """Extrai texto de um arquivo HTML e retorna um Document.

        Remove as seguintes tags antes da extração:
        ``script``, ``style``, ``nav``, ``footer``, ``header``.

        Args:
            filepath: Caminho absoluto para o arquivo .html.

        Returns:
            Document com texto limpo (sem tags) e metadata preenchida.
        """
        logger.info(f"Parsing HTML: {filepath}")
        content = Path(filepath).read_text(encoding="utf-8")
        soup = BeautifulSoup(content, "html.parser")

        # Remove elementos de estrutura e navegação — não são conteúdo semântico
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        doc_id = self._generate_doc_id(filepath)
        metadata = self._build_metadata(filepath, file_type="html")

        return Document(doc_id=doc_id, source_uri=filepath, text=text, metadata=metadata)
