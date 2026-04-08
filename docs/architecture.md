# Arquitetura do RAG System

## Visão Geral

O sistema é dividido em dois pipelines independentes que compartilham os mesmos contratos de dados — `Document` e `Chunk` — definidos em `core/models.py`. Essa separação é intencional: indexação e recuperação têm ciclos de vida distintos (indexação ocorre offline, recuperação ocorre em tempo real) e devem poder evoluir de forma independente.

```
Pipeline 1 — Indexação (offline)
  Arquivo → Parser → Cleaner → Chunker → BGE-M3 → ChromaDB
                                                 → BM25 Index

Pipeline 2 — Recuperação e Geração (online)
  Query → BGE-M3 → Busca Híbrida → BGE Reranker → PromptBuilder → Groq / Ollama → Resposta
```

***

## Contratos de Dados

Todos os componentes se comunicam exclusivamente por meio de dois modelos Pydantic definidos em `core/models.py`. Nenhum componente conhece detalhes de implementação de outro — apenas os contratos.

### `Document`

| Campo | Tipo | Descrição |
|---|---|---|
| `doc_id` | `str` | SHA256 (8 chars) do `source_uri` — garante idempotência na re-indexação |
| `source_uri` | `str` | Caminho absoluto do arquivo original |
| `text` | `str` | Texto extraído e limpo |
| `metadata` | `dict` | `filename`, `filetype`, e campos extras do parser (e.g., `sheet_name` para XLSX) |

### `Chunk`

| Campo | Tipo | Descrição |
|---|---|---|
| `chunk_id` | `str` | `{doc_id}_chunk_{index:04d}` — determinístico, idempotente no ChromaDB |
| `doc_id` | `str` | Referência ao `Document` de origem |
| `text` | `str` | Texto do trecho |
| `metadata` | `dict` | Herda do `Document` + `chunk_index` |
| `embedding` | `list[float] \| None` | Preenchido pelo `BGEEmbedder`; `None` antes da indexação |

**Por que `chunk_id` determinístico?** O `upsert` do ChromaDB é idempotente: re-indexar o mesmo arquivo não cria duplicatas. Isso permite re-rodar `make ingest` sem necessidade de `make reset` primeiro.

***

## Pipeline 1 — Indexação

### Parsing

**Decisão:** Usar Docling para PDF/DOCX, pandas para XLSX/CSV, e BeautifulSoup para HTML, coordenados por uma Factory (`parsers/factory.py`) que resolve o parser correto pela extensão do arquivo.

**Por que Docling para PDF?** PDFs são estruturalmente complexos: colunas, tabelas embutidas, cabeçalhos de página, figuras com legenda. Docling foi projetado especificamente para preservar essa estrutura, enquanto alternativas como `pdfminer` ou `PyPDF2` fazem extração linear que destrói o layout. Para um sistema RAG que precisa de chunks semanticamente coerentes, a qualidade do parsing é o fator mais crítico de toda a pipeline.

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| `PyPDF2` | Extração linear — destrói tabelas e colunas |
| `pdfminer.six` | Mais baixo nível que Docling, sem suporte a layout |
| LLM para parsing | Custo e latência inaceitáveis para documentos longos |
| LangChain `DocumentLoader` | Abstração excessiva; dificulta controle fino sobre metadados |

**Extensibilidade:** Adicionar suporte a `.pptx` requer apenas criar `pptx_parser.py` herdando de `ParserBase` e registrar na `PARSER_MAP`. Nenhum outro componente precisa ser alterado.

### Limpeza (`TextCleaner`)

Remove artefatos de extração: múltiplos espaços em branco, quebras de linha excessivas, caracteres de controle, e normaliza unicode. Opera sobre o `Document` completo antes do chunking — garantindo que o modelo de embedding nunca veja ruído tipográfico.

### Chunking

**Decisão:** Dois chunkers paralelos — `ParagraphChunker` (texto corrido) e `TableChunker` (tabelas detectadas pelo Docling).

**Por que chunkers separados?** Tabelas têm semântica fundamentalmente diferente do texto narrativo. Incluir uma tabela no meio de um chunk de parágrafo quebra o contexto dos dois. O `TableChunker` preserva tabelas como chunks atômicos, garantindo que o reranker possa avaliar a relevância da tabela como unidade.

**Parâmetros de chunking:**

| Parâmetro | Default | Raciocínio |
|---|---|---|
| `CHUNK_SIZE` | 512 chars | Cabe dentro da janela de contexto do BGE-M3 (8192 tokens), mantendo granularidade suficiente para precisão na recuperação |
| `CHUNK_OVERLAP` | 102 chars (~20%) | Evita perda de contexto em fronteiras de chunk sem duplicar excessivamente o índice |

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| Chunking por tamanho fixo | Quebra no meio de frases e parágrafos |
| Chunking por sentença (spaCy/NLTK) | Chunks muito pequenos → recuperação de baixa cobertura |
| Chunking semântico (LLM-based) | Custo e latência inaceitáveis em tempo de indexação |

### Embedding (`BGEEmbedder`)

**Modelo:** `BAAI/bge-m3`

**Por que BGE-M3?**
- Suporte nativo a português sem fine-tuning adicional
- Janela de contexto de 8192 tokens (vs. 512 tokens do `sentence-transformers/all-MiniLM`)
- Saída L2-normalizada nativamente → distância de cosseno equivale a produto interno (mais eficiente no ChromaDB)
- Benchmark MTEB: top-3 em tasks de recuperação cross-lingual

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| `all-MiniLM-L6-v2` | 384 dims, sem otimização para PT-BR |
| `text-embedding-3-small` (OpenAI) | Custo por chamada; dependência de API externa |
| `multilingual-e5-large` | Performance ligeiramente inferior ao BGE-M3 em MTEB PT-BR |

### Vector Store (`ChromaStore`)

**Decisão:** ChromaDB com persistência em disco, métrica cosseno, collection única.

**Por que ChromaDB?**
- Persistência nativa sem servidor dedicado (arquivo local)
- API Python simples e bem documentada
- Adequado para o volume de dados do projeto (dezenas a centenas de documentos)

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| Qdrant | Overhead operacional (servidor Docker separado) injustificado para este volume |
| Pinecone | SaaS — dependência de infraestrutura externa, custo |
| FAISS | Sem suporte nativo a metadados e filtragem; requer gestão manual do índice |
| pgvector | Requer PostgreSQL — stack adicional |

**Extensibilidade:** Trocar para Qdrant ou Pinecone requer apenas implementar `ChromaStore`-like herdando de `VectorStoreBase` em `core/interfaces.py`. O pipeline não precisa ser alterado.

### Índice BM25 (`BM25Search`)

O índice BM25 é construído com `rank-bm25` (algoritmo `BM25Okapi`) após cada ingesto e persistido em `bm25_index.pkl`. Na inicialização da API, o índice é carregado do disco — evitando reconstrução a cada restart.

**Por que persistir em pickle?** O índice BM25 é leve (centenas de KB para documentos típicos) e determinístico dado o mesmo corpus. Serializar junto com os objetos `Chunk` permite reconstrução exata dos resultados na recuperação sem re-acessar o ChromaDB.

***

## Pipeline 2 — Recuperação e Geração

### Busca Híbrida (`HybridSearch`)

A busca semântica via ChromaDB captura similaridade de significado mas falha em termos exatos (siglas, nomes próprios, identificadores). O BM25 complementa exatamente esses casos. Os scores são normalizados para  e combinados com pesos configuráveis:

```
score_final = α × score_semântico + (1 - α) × score_BM25
```

onde `α = HYBRID_SEMANTIC_WEIGHT` (default: 0.7).

**Por que 70/30?** Em domínios com documentos técnicos e CVs (uso primário deste sistema), a busca semântica produz recall mais alto. O BM25 tem peso menor mas garante que termos exatos como "Python", "RAGAS", ou nomes próprios não sejam ignorados por similaridade semântica fraca.

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| Apenas busca vetorial | Falha em termos exatos e siglas |
| Apenas BM25 | Sem captura de semântica; sinônimos não são recuperados |
| RRF (Reciprocal Rank Fusion) | Requer tuning adicional; pesos explícitos são mais interpretáveis para este projeto |

### Reranking (`BGEReranker`)

**Modelo:** `BAAI/bge-reranker-base` (cross-encoder)

**Por que cross-encoder?**

Bi-encoders (como o BGE-M3) calculam embeddings de query e chunk separadamente e medem similaridade por distância vetorial. Cross-encoders concatenam query e chunk e calculam um score de relevância conjunto — capturando interações semânticas que bi-encoders perdem.

O custo computacional é $$O(n)$$ em relação ao número de candidatos, enquanto bi-encoders são $$O(1)$$ por lookup vetorial. Por isso o reranker é aplicado apenas no top-20 da busca híbrida, reduzindo para top-5 com máxima precisão.

```
Busca Híbrida → top-20 candidatos → BGE Reranker → top-5 final
```

**Alternativas descartadas:**

| Alternativa | Motivo de descarte |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Treinado primariamente em inglês |
| LLM como reranker (GPT-4o) | Custo e latência proibitivos; overkill para este escopo |
| Sem reranker | Experimentos internos mostraram queda de precisão nos top-5 chunks |

### Geração (`GroqLLM` / `OllamaLLM`)

**LLM Principal:** `llama-3.1-8b-instant` via Groq API

**LLM Fallback:** `llama3.2:3b` via Ollama local

**Estratégia de fallback automático (`llm_provider=auto`):**

```python
1. Tenta Ollama (gratuito, local, sem latência de rede)
2. Se indisponível → Groq (gratuito com rate limits, remoto)
3. Se ambos indisponíveis → RuntimeError explícito
```

**Por que Groq como principal (não Ollama)?** O default é `llm_provider=groq` para garantir respostas consistentes em qualquer ambiente. Ollama requer que o servidor esteja rodando e o modelo baixado — condições que nem sempre são verdadeiras. O modo `auto` é para ambientes controlados onde Ollama está disponível.

**Por que `llama-3.1-8b-instant`?** Melhor relação custo/performance no tier gratuito do Groq para português. Modelos maiores (70B) ficam atrás de rate limits agressivos; 8B entrega qualidade suficiente para RAG com temperatura 0.1.

***

## Decisões de Arquitetura

### 1. Interfaces Abstratas no `core/`

Todos os componentes herdam de ABCs definidas em `core/interfaces.py` (`ParserBase`, `ChunkerBase`, `BaseLLM`, `VectorStoreBase`). Isso garante que qualquer componente pode ser substituído sem alterar o restante do sistema. O pipeline não conhece `ChromaDB` — conhece apenas `VectorStoreBase`.

### 2. Factory Pattern nos Parsers

`parsers/factory.py` recebe um `filepath`, lê a extensão e retorna o parser correto via `PARSER_MAP`. Adicionar `.pptx` = criar `pptx_parser.py` + uma linha no mapa. Zero alterações no pipeline.

### 3. `chunk_id` Determinístico

Gerado como `{doc_id}_chunk_{index:04d}`. O `upsert` do ChromaDB é idempotente: re-indexar não cria duplicatas. Isso torna o sistema resiliente a re-execuções acidentais de `make ingest`.

### 4. Singleton por Rota na API

Os componentes pesados (`BGEEmbedder`, `ChromaStore`, `RAGPipeline`) são inicializados uma única vez na primeira requisição via `global` nas rotas FastAPI. Reinicializar a cada request seria inaceitável: o BGE-M3 demora ~3s para carregar em CPU.

### 5. Separação Indexação / Recuperação

`run_ingestion.py` é um script standalone — não depende da API estar rodando. A API de recuperação (`/query`) não conhece nada sobre parsers ou chunkers. Essa separação permite atualizar o índice sem downtime da API.

***

## Contexto de Hardware e Requisitos de Execução

### Modelos Carregados em Memória

| Modelo | Tamanho em Disco | RAM (CPU) | GPU (opcional) |
|---|---|---|---|
| `BAAI/bge-m3` | ~570 MB | ~1.2 GB | ~600 MB VRAM |
| `BAAI/bge-reranker-base` | ~280 MB | ~600 MB | ~300 MB VRAM |
| **Total** | **~850 MB** | **~1.8 GB** | **~900 MB VRAM** |

### Configuração Mínima Recomendada

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | 4 cores | 8 cores |
| Disco | 2 GB | 5 GB |
| GPU | Não obrigatória | NVIDIA 4 GB+ VRAM |

**Nota sobre GPU:** `sentence-transformers` detecta automaticamente CUDA/MPS. Em CPU, a geração de embeddings para um documento de 10 páginas leva ~8s. Com GPU, < 1s.

### Tempos de Resposta Esperados (CPU, sem GPU)

| Operação | Tempo aproximado |
|---|---|
| Inicialização da API (cold start) | 8–12s |
| Ingesto de 1 documento PDF (10 págs) | 15–30s |
| Query completa (busca + rerank + LLM) | 3–6s |
| Query com Ollama local (LLM) | 8–15s |

***

## Configuração

Todos os parâmetros são centralizados em `core/config.py` via `pydantic-settings` e sobrescrevíveis pelo `.env`.

| Parâmetro | Default | Descrição |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Modelo de embedding |
| `EMBEDDING_BATCH_SIZE` | `16` | Chunks processados por batch no BGE-M3 |
| `CHUNK_SIZE` | `512` | Tamanho do chunk em caracteres |
| `CHUNK_OVERLAP` | `102` | Overlap entre chunks (~20%) |
| `VECTOR_SEARCH_TOP_K` | `20` | Candidatos da busca vetorial |
| `HYBRID_SEMANTIC_WEIGHT` | `0.7` | Peso da busca semântica na fusão |
| `HYBRID_BM25_WEIGHT` | `0.3` | Peso da busca BM25 na fusão |
| `RERANKER_TOP_K` | `5` | Chunks finais após reranking |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Modelo cross-encoder |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Modelo do Groq |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modelo do Ollama |
| `LLM_TEMPERATURE` | `0.1` | Temperatura do LLM (baixa = respostas mais determinísticas) |
| `CHROMADB_DIR` | `./chroma_db` | Diretório de persistência do ChromaDB |
| `CHROMA_COLLECTION_NAME` | `rag_system` | Nome da coleção no ChromaDB |

***

## Avaliação com RAGAS

O sistema inclui uma pipeline de avaliação automatizada em `evaluation/` usando o framework RAGAS 0.4.x.

### Métricas Avaliadas

| Métrica | O que mede |
|---|---|
| `faithfulness` | A resposta é fiel ao contexto recuperado? (sem alucinações) |
| `answer_relevancy` | A resposta é relevante para a pergunta feita? |
| `context_precision` | Os chunks recuperados contêm a informação necessária? |
| `context_recall` | O pipeline recuperou todos os chunks relevantes disponíveis? |

### Fluxo de Avaliação

```
make ingest-all
    ↓
make generate-tests     ← Gera casos de teste a partir dos chunks indexados
    ↓                      (LLM sintetiza perguntas e ground truth)
make eval               ← Roda RAGAS sobre os casos gerados
    ↓
evaluation/results/evaluation_results_*.json
```

### Configuração da Avaliação

O RAGAS usa um modelo LLM próprio para calcular as métricas — configurado separadamente do LLM de recuperação para evitar conflito de rate limits:

| Parâmetro | Default | Descrição |
|---|---|---|
| `RAGAS_GROQ_MODEL` | `llama-3.3-70b-versatile` | Modelo do Groq para avaliação RAGAS |
| `RAGAS_MAX_CHUNKS` | `3` | Chunks avaliados por execução |
| `RAGAS_MAX_CHUNK_CHARS` | `500` | Tamanho máximo de chunk para geração de casos |

Os resultados de cada execução são persistidos em `evaluation/results/` com timestamp — permitindo comparação entre versões do sistema.

***

## Estrutura de Pacotes

```
rag_system/
├── core/               # Contratos, modelos e configuração (sem dependências internas)
│   ├── interfaces.py   # ABCs: ParserBase, ChunkerBase, BaseLLM, VectorStoreBase
│   ├── models.py       # Document, Chunk (Pydantic)
│   ├── config.py       # Settings via pydantic-settings
│   └── logger.py       # Logger centralizado
├── ingestion/          # Pipeline 1 — indexação offline
│   ├── parsers/        # Factory + implementações por formato
│   ├── cleaners/       # TextCleaner
│   ├── chunkers/       # ParagraphChunker, TableChunker
│   ├── embedders/      # BGEEmbedder
│   └── vectorstore/    # ChromaStore
├── retrieval/          # Pipeline 2 — recuperação e geração online
│   ├── search/         # VectorSearch, BM25Search, HybridSearch
│   ├── reranker/       # BGEReranker
│   ├── generator/      # GroqLLM, OllamaLLM, PromptBuilder, LLMBase
│   └── pipeline.py     # RAGPipeline — orquestrador central
├── api/                # Interface HTTP (FastAPI)
│   ├── main.py
│   └── routes/
│       ├── query.py    # POST /query
│       └── ingest.py   # POST /ingest
└── evaluation/         # Avaliação RAGAS
    ├── generate_test_cases.py
    ├── ragas_eval.py
    └── results/
```