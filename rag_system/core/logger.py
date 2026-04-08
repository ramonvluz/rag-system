"""Fábrica de loggers padronizados para o RAG System.

Todos os módulos do projeto devem obter seus loggers exclusivamente
através de get_logger(__name__), garantindo formato e destinos
de saída consistentes em todo o sistema.

Uso:
    from rag_system.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Mensagem informativa")
    logger.debug("Detalhe para depuração")
"""

import logging
import sys
from rag_system.core.config import settings


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado com saída em console e arquivo.

    Implementa idempotência via verificação de handlers — chamar
    get_logger com o mesmo name múltiplas vezes retorna o mesmo
    logger sem duplicar handlers.

    Níveis de log por destino:
        - Console (stdout): INFO e acima
        - Arquivo (rag_system.log): DEBUG e acima

    Args:
        name: Nome do logger, convencionalmente ``__name__`` do módulo.

    Returns:
        Logger configurado e pronto para uso.
    """
    logger = logging.getLogger(name)

    # Evita duplicar handlers se o logger já foi configurado
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de console — apenas INFO e acima para não poluir o terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Handler de arquivo — DEBUG e acima para rastreabilidade completa
    log_file = settings.base_dir / "logs" / "rag_system.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger