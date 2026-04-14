#!/bin/bash
# scripts/add_headers.sh — Insere etiqueta de caminho na primeira linha de cada .py

find rag_system -name "*.py" | sort | while read filepath; do
    expected="# ${filepath}"
    first_line=$(head -n 1 "$filepath")

    if [ "$first_line" = "$expected" ]; then
        echo "⏭  Já existe: $filepath"
    else
        tmpfile=$(mktemp)
        printf "%s\n\n" "$expected" | cat - "$filepath" > "$tmpfile"
        mv "$tmpfile" "$filepath"
        echo "✅ Adicionado: $filepath"
    fi
done

echo ""
echo "✅ Concluído."