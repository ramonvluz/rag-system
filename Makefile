.PHONY: help setup start run run-fast run-quality run-ollama ingest ingest-all \
        eval eval-fast eval-quality generate-tests \
        test test-unit test-integration test-cov \
        lint clean reset snapshot

FILE ?= ""

help:
	@echo ""
	@echo "╔══════════════════════════════════════════╗"
	@echo "║         RAG System — Comandos             ║"
	@echo "╚══════════════════════════════════════════╝"
	@echo ""
	@echo "  make setup            Primeiro uso: cria .env e instala dependências"
	@echo "  make start            Indexa todos os documentos e sobe a API"
	@echo ""
	@echo "  make run               Inicia a API com modelo do .env"
	@echo "  make run-fast          API com llama-3.1-8b-instant (free tier)"
	@echo "  make run-quality       API com llama-3.3-70b-versatile (qualidade)"
	@echo "  make run-ollama        API com Ollama local"
	@echo ""
	@echo "  make ingest FILE=...   Indexa um arquivo específico"
	@echo "  make ingest-all        Indexa todos os arquivos em data/raw/"
	@echo "  make generate-tests    Gera casos de teste a partir dos chunks indexados"
	@echo ""
	@echo "  make eval              Avaliação RAGAS com modelo do .env"
	@echo "  make eval-fast         Avaliação com llama-3.1-8b-instant (free tier)"
	@echo "  make eval-quality      Avaliação com llama-3.3-70b-versatile (qualidade)"
	@echo ""
	@echo "  make test-unit         Testes unitários (rápido, sem serviços externos)"
	@echo "  make test-integration  Testes de integração (ChromaDB em tmpdir)"
	@echo "  make test              Roda todos os testes"
	@echo "  make test-cov          Todos os testes + relatório de cobertura HTML"
	@echo ""
	@echo "  make lint              Verifica qualidade do código"
	@echo "  make snapshot          Gera snapshot do projeto"
	@echo "  make clean             Remove arquivos temporários"
	@echo "  make reset             ⚠️  Limpa índices e logs (mantém data/raw/)"
	@echo ""
	@echo "  Fluxo após reset:"
	@echo "    make reset && make ingest-all && make generate-tests && make eval"
	@echo ""

# Setup & Start

setup:
	cp --update=none .env.example .env || true
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt || .venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup concluído."
	@echo "   Edite o .env com sua GROQ_API_KEY e rode: source .venv/bin/activate && make start"

start: ingest-all run

# API

run:
	uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload

run-fast:
	GROQ_MODEL=llama-3.1-8b-instant uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload

run-quality:
	GROQ_MODEL=llama-3.3-70b-versatile uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ollama:
	LLM_PROVIDER=ollama uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload

# Ingestion

ingest:
	python -m rag_system.ingestion.run_ingestion --file $(FILE)

ingest-all:
	python -m rag_system.ingestion.run_ingestion --all

# Avaliação RAGAS

generate-tests:
	python -m rag_system.evaluation.generate_test_cases --chunks 10 --per-chunk 2

eval:
	python -m rag_system.evaluation.ragas_eval

eval-fast:
	RAGAS_GROQ_MODEL=llama-3.1-8b-instant python -m rag_system.evaluation.ragas_eval

eval-quality:
	RAGAS_GROQ_MODEL=llama-3.3-70b-versatile python -m rag_system.evaluation.ragas_eval

# Testes

test-unit:
	pytest rag_system/tests/unit/ -v --tb=short

test-integration:
	pytest rag_system/tests/integration/ -v --tb=short

test: test-unit test-integration

test-cov:
	pytest rag_system/tests/ -v --tb=short \
	  --cov=rag_system --cov-report=term-missing --cov-report=html:htmlcov
	@echo ""
	@echo "📊 Relatório HTML gerado em htmlcov/index.html"

# Qualidade

lint:
	ruff check rag_system/

snapshot:
	bash scripts/generate_snapshot.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✅ Limpeza concluída."

reset:
	@echo "⚠️  Resetando o RAG System..."
	rm -rf chroma_db/
	rm -f rag_system/ingestion/ingestion_log.jsonl
	rm -f rag_system/evaluation/test_cases.jsonl
	rm -rf rag_system/evaluation/results/
	rm -f logs/rag_system.log
	find data/processed/ -type f ! -name ".gitkeep" -delete
	@echo "✅ Reset concluído."
	@echo "   Próximos passos: make ingest-all → make generate-tests → make eval"