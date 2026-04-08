# rag_system/evaluation/generate_test_cases.py
"""Gerador automático de casos de teste para avaliação RAGAS.

Usa o Groq como LLM de geração para criar pares (question, ground_truth)
a partir de chunks reais indexados no ChromaDB. O resultado é salvo
em test_cases.jsonl e consumido pelo runner de avaliação.

Execução:
    python -m rag_system.evaluation.generate_test_cases
    python -m rag_system.evaluation.generate_test_cases --chunks 20 --per-chunk 3
    ou: make generate-tests
"""

import json
from pathlib import Path

from groq import Groq

from rag_system.ingestion.vector_store.chroma_store import ChromaStore
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)

GENERATION_PROMPT = """Você é um avaliador de sistemas RAG. Com base no trecho de texto abaixo, gere {n} perguntas e suas respectivas respostas esperadas.

TEXTO:
{context}

INSTRUÇÕES:
- As perguntas devem ser objetivas e respondíveis apenas com base no texto
- As respostas (ground_truth) devem ser completas mas concisas
- Retorne APENAS um JSON válido, sem texto adicional, no formato:
[
  {{"question": "...", "ground_truth": "..."}},
  {{"question": "...", "ground_truth": "..."}}
]"""


def get_sample_chunks(n_chunks: int = 10) -> list[str]:
    """Busca chunks representativos do ChromaDB para geração de casos.

    Usa limit= diretamente na collection — não faz busca semântica,
    apenas amostra os primeiros n chunks por ordem de inserção.

    Args:
        n_chunks: Número de chunks a amostrar.

    Returns:
        Lista de textos dos chunks, ou lista vazia se o ChromaDB estiver vazio.
    """
    store = ChromaStore()
    results = store._collection.get(limit=n_chunks, include=["documents"])
    return results["documents"]


def generate_cases_from_chunk(client: Groq, chunk_text: str, n: int = 2) -> list[dict]:
    """Gera pares (question, ground_truth) a partir de um chunk via Groq.

    Args:
        client: Cliente Groq inicializado com a API key.
        chunk_text: Texto do chunk usado como contexto de geração.
        n: Número de pares pergunta/resposta a gerar por chunk.

    Returns:
        Lista de dicts com chaves ``question`` e ``ground_truth``.
        Retorna lista vazia em caso de falha.
    """
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{
                "role": "user",
                "content": GENERATION_PROMPT.format(context=chunk_text, n=n),
            }],
            temperature=0.3,
        )
        # ✅ BUG 1 CORRIGIDO: choices é lista, precisa do índice [0]
        content = response.choices[0].message.content.strip()

        # ✅ BUG 2 CORRIGIDO: split() retorna lista, tratamento correto do markdown
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]

        cases = json.loads(content)
        return [c for c in cases if "question" in c and "ground_truth" in c]

    except Exception as e:
        logger.warning(f"Falha ao gerar casos para chunk: {e}")
        return []


def generate_test_cases(
    n_chunks: int = 10,
    cases_per_chunk: int = 2,
    output_path: str = None,
) -> list[dict]:
    """Orquestra a geração completa de casos de teste.

    Fluxo:
        1. Amostra n_chunks do ChromaDB
        2. Para cada chunk, gera cases_per_chunk pares via Groq
        3. Deduplica por pergunta (case-insensitive)
        4. Salva em JSONL — uma linha por caso

    Args:
        n_chunks: Número de chunks a usar como base de geração.
        cases_per_chunk: Número de perguntas a gerar por chunk.
        output_path: Caminho do arquivo .jsonl de saída.
            Default: ``evaluation/test_cases.jsonl``.

    Returns:
        Lista deduplicada de dicts com ``question`` e ``ground_truth``.
    """
    logger.info(f"Gerando casos de teste: {n_chunks} chunks × {cases_per_chunk} perguntas...")

    client = Groq(api_key=settings.groq_api_key)
    chunks = get_sample_chunks(n_chunks)

    if not chunks:
        logger.error("Nenhum chunk encontrado no ChromaDB. Rode make ingest primeiro.")
        return []

    all_cases = []
    for i, chunk_text in enumerate(chunks):
        logger.info(f"Processando chunk {i + 1}/{len(chunks)}...")
        cases = generate_cases_from_chunk(client, chunk_text, cases_per_chunk)
        all_cases.extend(cases)
        logger.info(f"  → {len(cases)} casos gerados")

    # Deduplica por pergunta preservando ordem de aparição
    seen: set[str] = set()
    unique_cases = []
    for case in all_cases:
        q = case["question"].strip().lower()
        if q not in seen:
            seen.add(q)
            unique_cases.append(case)

    output = Path(output_path) if output_path else Path(__file__).parent / "test_cases.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for case in unique_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    logger.info(f"✅ {len(unique_cases)} casos salvos em '{output}'")
    return unique_cases


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Gera test_cases.jsonl automaticamente")
    arg_parser.add_argument("--chunks", type=int, default=10, help="Número de chunks a usar")
    arg_parser.add_argument("--per-chunk", type=int, default=2, help="Perguntas por chunk")
    arg_parser.add_argument("--output", type=str, default=None, help="Caminho do arquivo de saída")
    args = arg_parser.parse_args()

    cases = generate_test_cases(
        n_chunks=args.chunks,
        cases_per_chunk=args.per_chunk,
        output_path=args.output,
    )
    print(f"\n✅ Total de casos gerados: {len(cases)}")