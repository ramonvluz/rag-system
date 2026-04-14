# rag_system/evaluation/ragas_eval.py

"""Runner de avaliação RAGAS com Groq como LLM avaliador.

Executa o pipeline RAG sobre cada caso de teste do test_cases.jsonl,
monta um EvaluationDataset e avalia com as métricas Faithfulness,
AnswerRelevancy e ContextPrecision. Resultados são salvos em
evaluation/results/ como JSON timestampado.

Execução:
    python -m rag_system.evaluation.evaluator
    ou: make evaluate
"""

import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory

from rag_system.retrieval.pipeline import RAGPipeline
from rag_system.evaluation.metrics import load_test_cases, print_scores
from rag_system.core.config import settings
from rag_system.core.logger import get_logger

logger = get_logger(__name__)

# Diretório de saída para resultados timestampados
RESULTS_DIR = Path(__file__).parent / "results"


def _truncate_contexts(chunks: list, max_chunks: int, max_chars: int) -> list[str]:
    """Limita quantidade e tamanho dos chunks enviados ao RAGAS.

    Necessário para evitar erros de limite de tokens na API Groq
    durante a avaliação — ragas_max_chunks e ragas_max_chunk_chars
    são configuráveis via .env.

    Args:
        chunks: Lista de objetos Chunk do resultado do pipeline.
        max_chunks: Número máximo de chunks a incluir por amostra.
        max_chars: Número máximo de caracteres por chunk.

    Returns:
        Lista de strings truncadas prontas para o SingleTurnSample.
    """
    return [chunk.text[:max_chars] for chunk in chunks[:max_chunks]]


def build_ragas_config():
    """Configura o LLM e embeddings para o avaliador RAGAS.

    Usa o cliente OpenAI apontando para a base URL do Groq — compatível
    com a interface OpenAI que o ragas.llms.llm_factory espera.
    Embeddings reutilizam o mesmo modelo BGE-M3 da ingestão para
    consistência na métrica AnswerRelevancy.

    Returns:
        Tupla (llm, embeddings) prontos para uso nas métricas RAGAS.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings as LCEmbeddings

    # Groq expõe API compatível com OpenAI — reutilizado como avaliador RAGAS
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    llm = llm_factory(settings.ragas_groq_model, client=client)

    # Mesmo modelo de embeddings da ingestão — garante espaço vetorial consistente
    embeddings = LangchainEmbeddingsWrapper(
        LCEmbeddings(model_name=settings.embedding_model)
    )
    return llm, embeddings


def run_evaluation() -> dict:
    """Executa a avaliação RAGAS completa sobre todos os casos de teste.

    Fluxo:
        1. Carrega test_cases.jsonl
        2. Para cada caso, executa pipeline.query() e coleta chunks
        3. Monta EvaluationDataset com SingleTurnSample
        4. Avalia com Faithfulness, AnswerRelevancy e ContextPrecision
        5. Exibe scores no terminal e persiste em results/

    Returns:
        Dicionário com scores médios por métrica.
    """
    logger.info("Iniciando avaliação RAGAS com Groq...")
    pipeline = RAGPipeline(llm_provider=settings.llm_provider)
    test_cases = load_test_cases()
    samples = []

    for item in test_cases:
        question = item["question"]
        logger.info(f"Avaliando: '{question}'")
        result = pipeline.query(question)

        samples.append(SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=_truncate_contexts(
                result["chunks"],
                max_chunks=settings.ragas_max_chunks,
                max_chars=settings.ragas_max_chunk_chars,
            ),
            reference=item["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)
    llm, embeddings = build_ragas_config()

    scores = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
        ],
    )

    # Extrai médias das 3 métricas do DataFrame retornado pelo RAGAS
    scores_dict = scores.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision"]
    ].mean().to_dict()

    print_scores(scores_dict)
    _save_results(scores_dict)
    return scores_dict


def _save_results(scores_dict: dict) -> Path:
    """Persiste os scores em um arquivo JSON timestampado.

    Cada execução gera um arquivo único em evaluation/results/,
    permitindo rastrear evolução dos scores ao longo do desenvolvimento.

    Args:
        scores_dict: Dicionário métrica → score médio.

    Returns:
        Path do arquivo JSON gerado.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RESULTS_DIR / f"evaluation_results_{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "scores": scores_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Resultados salvos em '{output_path}'")
    return output_path


if __name__ == "__main__":
    run_evaluation()