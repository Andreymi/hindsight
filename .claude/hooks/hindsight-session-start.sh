#!/bin/bash
# hindsight-session-start.sh
# Загружает релевантный контекст из Hindsight при старте сессии
#
# Stdout автоматически становится additionalContext для Claude

# === GRACEFUL DEGRADATION ===
# Никогда не блокируем Claude — любая ошибка = тихий exit
trap 'exit 0' ERR SIGTERM SIGINT

# Читаем входные данные
INPUT=$(cat) || exit 0
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null) || exit 0
SOURCE=$(echo "$INPUT" | jq -r '.source // "startup"' 2>/dev/null) || exit 0

# Только при startup или resume
[[ "$SOURCE" != "startup" && "$SOURCE" != "resume" ]] && exit 0

# Конфигурация
BANK_ID="${HINDSIGHT_BANK_ID:-hindsight-dev}"

# Используем локальный hindsight-embed из проекта
HINDSIGHT_EMBED="${HINDSIGHT_EMBED_PATH:-/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed}"

# Проверяем что hindsight-embed доступен
[[ ! -x "$HINDSIGHT_EMBED" ]] && exit 0

# Имя проекта из директории
PROJECT_NAME=$(basename "$CWD" 2>/dev/null || echo "unknown")

# Recall релевантного контекста
RESULT=$($HINDSIGHT_EMBED memory recall "$BANK_ID" \
    "project $PROJECT_NAME context preferences architecture decisions" \
    -b low --max-tokens 1500 -o json 2>/dev/null) || exit 0

# Извлекаем факты (API возвращает .results[], не .facts[])
# Берём только основную часть до первого " |" для краткости
FACTS=$(echo "$RESULT" | jq -r '.results[]?.text' 2>/dev/null | cut -d'|' -f1 | head -6)

# Если есть факты — выводим как контекст
if [[ -n "$FACTS" ]]; then
    echo "## Hindsight Memory"
    echo ""
    echo "Relevant context for '$PROJECT_NAME':"
    echo "$FACTS" | while IFS= read -r fact; do
        [[ -n "$fact" ]] && echo "- $fact"
    done
fi

exit 0
