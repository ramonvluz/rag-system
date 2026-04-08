## Gera um snapshot completo do código-fonte em um único arquivo .txt,
# otimizado para ser colado em assistentes de IA (ChatGPT, Claude, Gemini,
# Perplexity) para análise, pair programming, refatoração e code review.
#
# O snapshot captura:
#   - Estrutura de pastas do projeto (via tree)
#   - Todo o código-fonte Python (.py)
#   - Configurações (pyproject.toml, requirements.txt, Makefile, .env.example)
#   - Documentação (.md), scripts (.sh) e casos de teste (.jsonl)
#
# O snapshot EXCLUI automaticamente:
#   - Artefatos gerados (.venv, __pycache__, chroma_db, modelos, dados brutos)
#   - Segredos (.env)
#   - Snapshots anteriores
#
# Uso:
#   bash scripts/generate_snapshot.sh
#   make snapshot
#
# Output:
#   snapshot_YYYYMMDD_HHMM.txt (na raiz do projeto)

set -euo pipefail

OUTPUT="snapshot_$(date '+%Y%m%d_%H%M').txt"
PROJECT_DIR="."

TREE_IGNORE="__pycache__|*.pyc|*.pyo|.venv|chroma_db|*.egg-info|.git|node_modules|*.pkl|*.bin|*.safetensors|*.onnx|models|snapshot_*.txt|projeto_completo.txt"

echo "=== RAG SYSTEM — SNAPSHOT $(date '+%Y-%m-%d %H:%M') ===" > "$OUTPUT"
echo "" >> "$OUTPUT"

echo "========== ESTRUTURA ==========" >> "$OUTPUT"
tree "$PROJECT_DIR" \
  --noreport \
  -I "$TREE_IGNORE" \
  >> "$OUTPUT"
echo "" >> "$OUTPUT"

echo "========== ARQUIVOS ==========" >> "$OUTPUT"

find "$PROJECT_DIR" \
  -type f \
  \( \
    -name "*.py" \
    -o -name "*.toml" \
    -o -name "*.cfg" \
    -o -name "*.ini" \
    -o -name "*.sh" \
    -o -name "Makefile" \
    -o -name "*.md" \
    -o -name ".env.example" \
    -o -name ".gitignore" \
    -o -name "*.jsonl" \
    -o -name "requirements.txt" \
  \) \
  ! -path "*/.venv/*" \
  ! -path "*/__pycache__/*" \
  ! -path "*/.git/*" \
  ! -path "*/chroma_db/*" \
  ! -path "*/models/*" \
  ! -path "*/data/raw/*" \
  ! -path "*/data/processed/*" \
  ! -path "*/results/*" \
  ! -path "*/*.egg-info/*" \
  ! -path "*/.idea/*" \
  ! -path "*/.mypy_cache/*" \
  ! -path "*/.pytest_cache/*" \
  ! -name "ingestion_log.jsonl" \
  ! -name "snapshot_*.txt" \
  ! -name "projeto_completo.txt" \
  | sort \
  | while read -r file; do
      echo ""                      >> "$OUTPUT"
      echo "--- FILE: $file ---"   >> "$OUTPUT"
      cat "$file"                  >> "$OUTPUT"
      echo ""                      >> "$OUTPUT"
    done

echo "" >> "$OUTPUT"
echo "=== FIM DO SNAPSHOT ===" >> "$OUTPUT"

BYTES=$(wc -c < "$OUTPUT")
FILES=$(grep -c "^--- FILE:" "$OUTPUT" || true)

echo "✅ Snapshot gerado: $OUTPUT"
echo "   Arquivos capturados : $FILES"
echo "   Tamanho             : $BYTES bytes"