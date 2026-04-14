# rag_system/ingestion/cleaners/text_cleaner.py

"""Limpeza e normalização de texto extraído por parsers.

Aplica um pipeline sequencial de transformações para remover ruídos
comuns em textos extraídos de PDFs e documentos Office, preparando
o conteúdo para um chunking de maior qualidade.
"""

import re
import unicodedata

from rag_system.core.models import Document
from rag_system.core.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Pipeline de limpeza de texto para documentos extraídos por parsers.

    Aplica as transformações na seguinte ordem:
        1. Normalização Unicode (NFC)
        2. Remoção de caracteres de controle
        3. Correção de espaços e quebras de linha excessivas
        4. Remoção de cabeçalhos/rodapés repetidos
        5. Reconexão de palavras hifenizadas

    A ordem importa — normalizar Unicode antes de aplicar regex
    garante que padrões com ``\\w`` funcionem corretamente em PT-BR.
    """

    def clean(self, document: Document) -> Document:
        """Aplica o pipeline de limpeza e retorna um novo Document.

        Não modifica o Document original — retorna uma nova instância
        com o texto limpo e o flag ``cleaned: True`` nos metadados.

        Args:
            document: Document bruto, saída de qualquer parser.

        Returns:
            Novo Document com texto limpo e metadata atualizada.
        """
        logger.info(f"Limpando documento: {document.metadata.get('filename', document.doc_id)}")
        text = document.text
        text = self._normalize_unicode(text)
        text = self._remove_control_characters(text)
        text = self._fix_whitespace(text)
        text = self._remove_repeated_headers(text)
        text = self._fix_hyphenation(text)
        text = text.strip()

        logger.debug(f"Chars antes: {len(document.text)} | depois: {len(text)}")

        return Document(
            doc_id=document.doc_id,
            source_uri=document.source_uri,
            text=text,
            metadata={**document.metadata, "cleaned": True},
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normaliza unicode para a forma NFC.

        Essencial para PT-BR — caracteres acentuados podem ser
        representados como um único codepoint (NFC) ou como letra
        base + combining accent (NFD). A normalização garante
        consistência para os regex aplicados nas etapas seguintes.

        Args:
            text: Texto a normalizar.

        Returns:
            Texto com unicode normalizado para NFC.
        """
        return unicodedata.normalize("NFC", text)

    def _remove_control_characters(self, text: str) -> str:
        """Remove caracteres de controle invisíveis, preservando newlines e tabs.

        PDFs frequentemente introduzem caracteres de controle (form feed,
        null bytes, etc.) que corrompem o texto downstream.

        Args:
            text: Texto a limpar.

        Returns:
            Texto sem caracteres de controle, com espaço como substituto.
        """
        # Preserva \n e \t — remove qualquer outro whitespace não imprimível
        return re.sub(r"[^\S\n\t ]+", " ", text)

    def _fix_whitespace(self, text: str) -> str:
        """Normaliza espaços duplos e quebras de linha excessivas.

        Limita sequências de newlines a no máximo 2 seguidas,
        preservando separação visual entre parágrafos sem criar
        blocos em branco excessivos.

        Args:
            text: Texto a normalizar.

        Returns:
            Texto com no máximo um espaço consecutivo e
            no máximo duas quebras de linha consecutivas.
        """
        text = re.sub(r" {2,}", " ", text)          # Colapsa espaços múltiplos
        text = re.sub(r"\n{3,}", "\n\n", text)      # Limita linhas em branco a 2
        return text

    def _remove_repeated_headers(self, text: str) -> str:
        """Remove linhas idênticas repetidas mais de duas vezes.

        Cabeçalhos e rodapés de PDF (ex: nome do documento, número
        de página) tendem a se repetir em todas as páginas. Linhas
        que aparecem mais de 2 vezes no documento são descartadas.

        Args:
            text: Texto a filtrar.

        Returns:
            Texto sem linhas com repetição excessiva.
        """
        lines = text.split("\n")
        seen: dict[str, int] = {}
        cleaned = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Preserva linhas em branco — fazem parte da estrutura
                cleaned.append(line)
                continue
            seen[stripped] = seen.get(stripped, 0) + 1
            # Permite até 2 ocorrências — títulos de seção podem se repetir
            if seen[stripped] <= 2:
                cleaned.append(line)

        return "\n".join(cleaned)

    def _fix_hyphenation(self, text: str) -> str:
        """Reconecta palavras quebradas com hífen no fim da linha.

        PDFs com colunas estreitas frequentemente quebram palavras com
        hífen ao final da linha. Ex: ``infor-\\nmação`` → ``informação``.

        Args:
            text: Texto a corrigir.

        Returns:
            Texto com palavras hifenizadas reconectadas.
        """
        # Padrão: letra + hífen + newline + letra → remove hífen e newline
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)