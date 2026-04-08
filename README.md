# RAG System

Sistema de Recuperação e Geração Aumentada (RAG) de ponta a ponta, construído com Python puro e ferramentas de código aberto. Desenvolvido como projeto de portfólio para demonstrar arquitetura de sistemas de IA/ML em produção — projetado para rodar integralmente em CPU, sem GPU dedicada, com custo operacional zero.

---

## Visão Geral

O sistema responde perguntas sobre documentos indexados combinando busca semântica, busca lexical BM25 e reranking para selecionar o contexto mais relevante, que é então enviado a um LLM para geração da resposta final.

```
Documento → Parser → Cleaner → Chunker → BGE-M3 → ChromaDB
Pergunta  → BGE-M3 → Busca Híbrida → BGE Reranker → Prompt → Groq / Ollama → Resposta
```

---

## Stack

| Componente    | Tecnologia                          |
|---------------|-------------------------------------|
| Parsing       | Docling, BeautifulSoup, pandas      |
| Embedding     | BGE-M3 (`BAAI/bge-m3`)              |
| Vector Store  | ChromaDB                            |
| Busca Lexical | rank-bm25                           |
| Reranking     | BGE Reranker (cross-encoder)        |
| LLM Principal | Groq (`llama-3.1-8b-instant`)       |
| LLM Fallback  | Ollama (`llama3.2:3b`)              |
| API           | FastAPI + Uvicorn                   |
| Avaliação     | RAGAS 0.4.x                         |
| Testes        | pytest + pytest-cov                 |
| Qualidade     | ruff                                |

---

## Estrutura do Projeto

```
rag-system/
├── rag_system/
│   ├── api/                    # Interface HTTP (FastAPI)
│   │   ├── routes/
│   │   │   ├── query.py        # POST /query
│   │   │   └── ingest.py       # POST /ingest
│   │   ├── main.py
│   │   └── schemas.py
│   ├── core/                   # Contratos, modelos e configuração
│   │   ├── config.py           # Settings via pydantic-settings
│   │   ├── interfaces.py       # ABCs: BaseParser, BaseLLM, etc.
│   │   ├── models.py           # Dataclasses: Document, Chunk
│   │   └── logger.py
│   ├── ingestion/              # Pipeline de indexação (offline)
│   │   ├── parsers/            # PDF, DOCX, XLSX, CSV, HTML
│   │   ├── cleaners/           # Normalização de texto
│   │   ├── chunkers/           # ParagraphChunker, TableChunker
│   │   ├── embedders/          # BGEEmbedder (BGE-M3)
│   │   ├── vector_store/       # ChromaStore
│   │   └── run_ingestion.py    # Entrypoint CLI
│   ├── retrieval/              # Pipeline de recuperação (online)
│   │   ├── search/             # VectorSearch, BM25Search, HybridSearch
│   │   ├── reranker/           # BGEReranker (cross-encoder)
│   │   ├── generator/          # GroqLLM, OllamaLLM, PromptBuilder
│   │   └── pipeline.py         # Orquestrador RAGPipeline
│   ├── evaluation/             # Avaliação com RAGAS
│   │   ├── generate_test_cases.py
│   │   └── ragas_eval.py
│   └── tests/
│       ├── unit/               # 55 testes, sem serviços externos
│       └── integration/        # 48 testes, ChromaDB em tmpdir
├── data/
│   ├── raw/                    # Documentos originais a indexar
│   └── processed/              # JSONs pós-processamento
├── docs/
│   └── architecture.md         # Decisões técnicas aprofundadas
├── scripts/
│   └── generate_snapshot.sh
├── .env.example
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.ai) (opcional — necessário para `run-ollama`)
- Chave de API do [Groq](https://console.groq.com) (gratuita)

### Setup

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/rag-system.git
cd rag-system

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 3. Setup automático (cria .env e instala dependências)
make setup
# Edite o .env gerado e preencha GROQ_API_KEY
```

### Ollama (opcional)

```bash
# Instale em: https://ollama.ai
ollama pull llama3.2:3b
```

---

## Uso Rápido

### Início rápido (tudo em um comando)

```bash
make start   # indexa todos os arquivos em data/raw/ e sobe a API
```

Equivalente a `make ingest-all && make run`. Ideal para o primeiro uso
após colocar documentos em `data/raw/` e configurar o `.env`.

---

### 1. Indexar documentos

```bash
make ingest FILE=data/raw/documento.pdf   # arquivo único
make ingest-all                            # todos os arquivos em data/raw/
```

Formatos suportados: `.pdf`, `.docx`, `.xlsx`, `.csv`, `.html`

### 2. Iniciar a API

```bash
make run           # modelo definido no .env
make run-fast      # força llama-3.1-8b-instant (free tier)
make run-quality   # força llama-3.3-70b-versatile
make run-ollama    # força Ollama local
```

A API ficará disponível em `http://localhost:8000`.
Documentação interativa: `http://localhost:8000/docs`

### 3. Fazer uma pergunta

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual a experiência profissional em IA?"}'
```

```json
{
  "answer": "O candidato possui experiência em...",
  "sources": ["cv.pdf"]
}
```

### 4. Indexar via API

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/caminho/absoluto/documento.pdf"}'
```

---

## Fluxo Completo (após reset)

```bash
make reset          # limpa índices e logs, mantém data/raw/
make ingest-all     # indexa todos os documentos
make generate-tests # gera casos de teste a partir dos chunks
make eval           # roda avaliação RAGAS
make run            # inicia a API
```

---

## Workflow com IA

O projeto inclui um script para gerar um snapshot completo do código em um único
arquivo `.txt`, otimizado para ser colado em assistentes de IA (ChatGPT, Claude,
Gemini, Perplexity) para análise, refatoração ou pair programming.

```bash
make snapshot
# Gera: snapshot_YYYYMMDD_HHMM.txt (~arquivos capturados, tamanho em bytes)
```

O snapshot captura todo o código-fonte, configurações e documentação, excluindo
artefatos gerados (`.venv`, `chroma_db`, modelos, dados brutos).

---

## Avaliação RAGAS

```bash
make eval          # modelo definido em RAGAS_GROQ_MODEL no .env
make eval-fast     # llama-3.1-8b-instant (free tier, rápido)
make eval-quality  # llama-3.3-70b-versatile (máxima qualidade)
```

| Métrica           | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` |
|-------------------|------------------------|---------------------------|
| Faithfulness      | 0.7294                 | **0.8230**                |
| Answer Relevancy  | 0.7912                 | **0.8427**                |
| Context Precision | 0.7857                 | **0.9038**                |

> O modelo juiz influencia os scores — o `70b` é mais criterioso, especialmente
> em Context Precision (+11.8 pp). Ver `docs/architecture.md` para análise detalhada.

---

## Testes

```bash
make test-unit         # 55 testes unitários — rápido, sem serviços externos
make test-integration  # 48 testes de integração — ChromaDB em tmpdir
make test              # suite completa (unit + integration)
make test-cov          # suite completa + relatório de cobertura HTML
```

```
TOTAL    835 stmts    198 missed    76.29%
```

---

## Configuração

Todos os parâmetros são centralizados em `core/config.py` via `pydantic-settings`
e sobrescrevíveis por variável de ambiente sem alterar código.

| Parâmetro                | Default                    | Descrição                               |
|--------------------------|----------------------------|-----------------------------------------|
| `LLM_PROVIDER`           | `groq`                     | `groq` / `ollama` / `auto`              |
| `GROQ_MODEL`             | `llama-3.1-8b-instant`     | Modelo do Groq                          |
| `OLLAMA_MODEL`           | `llama3.2:3b`              | Modelo local do Ollama                  |
| `EMBEDDING_MODEL`        | `BAAI/bge-m3`              | Modelo de embedding                     |
| `RERANKER_MODEL`         | `BAAI/bge-reranker-base`   | Modelo cross-encoder                    |
| `CHUNK_SIZE`             | `768`                      | Tamanho máximo do chunk (caracteres)    |
| `CHUNK_OVERLAP`          | `200`                      | Overlap entre chunks                    |
| `VECTOR_SEARCH_TOP_K`    | `20`                       | Candidatos da busca vetorial            |
| `HYBRID_SEMANTIC_WEIGHT` | `0.7`                      | Peso da busca semântica na fusão        |
| `HYBRID_BM25_WEIGHT`     | `0.3`                      | Peso da busca BM25 na fusão             |
| `RERANKER_TOP_K`         | `5`                        | Chunks finais após reranking            |
| `RAGAS_GROQ_MODEL`       | `llama-3.1-8b-instant`     | Modelo juiz para avaliação RAGAS        |

Consulte `.env.example` para a lista completa.

---

## Endpoints da API

| Método | Rota      | Descrição                                      |
|--------|-----------|------------------------------------------------|
| `POST` | `/query`  | Faz uma pergunta ao sistema RAG                |
| `POST` | `/ingest` | Indexa um novo documento pelo caminho absoluto |
| `GET`  | `/health` | Verifica se a API está no ar                   |

---

## Decisões de Arquitetura

> O sistema foi projetado para rodar em CPU, sem GPU dedicada, com custo zero.
> Cada escolha reflete essa restrição. Para o raciocínio completo e comparação
> com alternativas, veja [`docs/architecture.md`](docs/architecture.md).

1. **Interfaces abstratas no `core`** — ABCs em `core/interfaces.py` permitem trocar ChromaDB → Qdrant ou Groq → OpenAI sem alterar o restante do sistema.
2. **Factory Pattern nos parsers** — `parsers/factory.py` seleciona o parser pela extensão. Adicionar `.pptx` é apenas criar o parser e registrar no `PARSER_MAP`.
3. **Busca Híbrida** — semântica (ChromaDB) + lexical (BM25) com pesos configuráveis (`0.7 / 0.3`). O BM25 cobre siglas e nomes próprios que a busca semântica perde.
4. **Reranking com Cross-Encoder** — BGE Reranker aplicado apenas no top-20, reduzindo para top-5. Mais preciso que bi-encoder, justifica o custo extra por ser seletivo.
5. **LLM com fallback automático** — `auto` tenta Ollama (local, zero custo) e cai para Groq se indisponível.
6. **`chunk_id` determinístico** — `{doc_id}_chunk{index:04d}` torna o upsert idempotente. Re-indexar o mesmo documento não cria duplicatas.

---

## Qualidade de Código

```bash
make lint    # ruff check rag_system/
make help    # lista todos os comandos disponíveis
```

---

## Licença

MIT