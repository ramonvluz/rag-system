# rag_system/ingestion/parsers/base.py

"""Classe base concreta para todos os parsers do RAG System.

Fornece helpers compartilhados de geração de doc_id e metadados,
eliminando duplicação entre PDFParser, DOCXParser, CSVParser, etc.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

from rag_system.core.interfaces import BaseParser
from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class ParserBase(BaseParser):
    """Classe base concreta para parsers — adiciona helpers compartilhados.

    Subclasses devem obrigatoriamente implementar parse(). Os métodos
    _generate_doc_id() e _build_metadata() são helpers prontos para uso,
    garantindo consistência de formato entre todos os parsers.
    """

    def parse(self, filepath: str) -> Document:
        """Lê e converte um arquivo em Document.

        Args:
            filepath: Caminho absoluto para o arquivo de origem.

        Returns:
            Document com text extraído e metadata preenchida.

        Raises:
            NotImplementedError: Sempre — subclasses devem implementar.
        """
        raise NotImplementedError("Subclasses devem implementar parse()")

    def _generate_doc_id(self, filepath: str) -> str:
        """Gera doc_id legível, único e determinístico.

        Combina o stem do arquivo com os primeiros 8 caracteres do hash
        MD5 do caminho completo — legível para humanos e único mesmo
        quando dois arquivos têm o mesmo nome em pastas diferentes.

        Args:
            filepath: Caminho do arquivo de origem.

        Returns:
            String no formato ``{stem}_{hash8}``.
            Exemplo: ``relatorio_2024_a3f82c11``
        """
        stem = Path(filepath).stem
        # Hash do caminho completo garante unicidade entre arquivos homônimos
        short_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        return f"{stem}_{short_hash}"

    def _build_metadata(self, filepath: str, file_type: str, **extra) -> dict:
        """Monta metadados padrão para qualquer documento.

        Os campos retornados são incluídos em Document.metadata e
        herdados por todos os Chunks gerados a partir do documento.

        Args:
            filepath: Caminho do arquivo de origem.
            file_type: Extensão sem ponto — ex: ``pdf``, ``docx``, ``csv``.
            **extra: Metadados adicionais específicos de cada parser
                — ex: ``rows=120``, ``sheets=["Plan1"]``.

        Returns:
            Dicionário com source_uri, filename, file_type,
            file_size_bytes, indexed_at e quaisquer campos extras.
        """
        p = Path(filepath)
        return {
            "source_uri": str(p.resolve()),
            "filename": p.name,
            "file_type": file_type,
            "file_size_bytes": p.stat().st_size if p.exists() else 0,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }