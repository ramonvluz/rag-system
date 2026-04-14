# rag_system/evaluation/metrics.py

"""Utilitários de métricas e apresentação para a avaliação RAGAS.

Fornece helpers compartilhados entre o runner de avaliação (evaluator.py)
e qualquer script externo que precise carregar casos de teste ou
exibir scores formatados no terminal.
"""

import json
import math
from pathlib import Path

from rag_system.core.logger import get_logger

logger = get_logger(__name__)


def load_test_cases(path: str = None) -> list[dict]:
    """Carrega os casos de teste do arquivo JSONL.

    Ignora linhas em branco — formato robusto para arquivos gerados
    incrementalmente pelo generate_test_cases.py.

    Args:
        path: Caminho do arquivo .jsonl. Default: ``evaluation/test_cases.jsonl``.

    Returns:
        Lista de dicts com chaves ``question`` e ``ground_truth``.

    Raises:
        FileNotFoundError: Se o arquivo não existir — indica que
            make generate-tests ainda não foi executado.
    """
    filepath = Path(path) if path else Path(__file__).parent / "test_cases.jsonl"
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    logger.info(f"{len(cases)} casos de teste carregados de '{filepath}'")
    return cases


def print_scores(scores: dict) -> None:
    """Exibe os scores RAGAS formatados no terminal com barra de progresso visual.

    Métricas com valor NaN (falha interna do RAGAS, geralmente por
    contexto vazio ou erro no LLM avaliador) são exibidas como ``N/A``
    em vez de quebrar a saída.

    Args:
        scores: Dicionário métrica → valor float, saída de
            ``scores_df.mean().to_dict()``.
    """
    print("\n" + "=" * 50)
    print("📊 RESULTADOS DA AVALIAÇÃO RAGAS")
    print("=" * 50)
    for metric, value in scores.items():
        if isinstance(value, float) and not math.isnan(value):
            # Barra visual proporcional — 20 blocos = score 1.0
            bar = "█" * int(value * 20)
            print(f"  {metric:<25} {value:.4f}  |{bar:<20}|")
        else:
            print(f"  {metric:<25} N/A (métrica falhou)")
    print("=" * 50 + "\n")