#!/bin/bash
# hindsight-user-prompt.sh
# Умный recall по контексту задачи пользователя
#
# Логика:
# 1. Фильтруем незначимые сообщения (короткие, "ок", "да" и т.д.)
# 2. Извлекаем ключевые слова из prompt
# 3. Делаем recall по теме
# 4. Возвращаем релевантный контекст

# === GRACEFUL DEGRADATION ===
# Никогда не блокируем Claude — любая ошибка = тихий exit
trap 'exit 0' ERR SIGTERM SIGINT

INPUT=$(cat) || exit 0
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty' 2>/dev/null) || exit 0

# Конфигурация
BANK_ID="${HINDSIGHT_BANK_ID:-hindsight-dev}"
MIN_PROMPT_LENGTH=30
MAX_FACTS=5

# Используем локальный hindsight-embed из проекта
HINDSIGHT_EMBED="${HINDSIGHT_EMBED_PATH:-/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed}"

# Проверяем что hindsight-embed доступен
[[ ! -x "$HINDSIGHT_EMBED" ]] && exit 0

# Пропускаем пустые промпты
[[ -z "$PROMPT" ]] && exit 0

# Пропускаем короткие сообщения
[[ ${#PROMPT} -lt $MIN_PROMPT_LENGTH ]] && exit 0

# Пропускаем тривиальные ответы (регистронезависимо)
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')
TRIVIAL_PATTERNS="^(ок|ok|да|yes|нет|no|спасибо|thanks|понял|got it|хорошо|good|ладно|fine|продолжай|continue|дальше|next)\.?!?$"
if echo "$PROMPT_LOWER" | grep -qE "$TRIVIAL_PATTERNS"; then
    exit 0
fi

# Пропускаем если это просто подтверждение или короткая команда
if echo "$PROMPT_LOWER" | grep -qE "^(y|n|1|2|3|4|q|quit|exit|help|\?)$"; then
    exit 0
fi

# Формируем запрос для recall — используем сам prompt как query
# hindsight сам извлечёт ключевые слова через query analyzer
RESULT=$($HINDSIGHT_EMBED memory recall "$BANK_ID" "$PROMPT" \
    -b low --max-tokens 1500 -o json 2>/dev/null) || exit 0

# Извлекаем факты
FACTS=$(echo "$RESULT" | jq -r '.results[]?.text' 2>/dev/null | cut -d'|' -f1 | head -$MAX_FACTS)

# Если нашли релевантные факты — возвращаем как контекст
if [[ -n "$FACTS" ]]; then
    # Формируем компактный контекст
    CONTEXT="## Relevant Hindsight Memory\n"
    while IFS= read -r fact; do
        [[ -n "$fact" ]] && CONTEXT="${CONTEXT}• ${fact}\n"
    done <<< "$FACTS"

    # JSON output с additionalContext
    jq -n --arg ctx "$CONTEXT" '{
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": $ctx
        }
    }'
fi

exit 0
