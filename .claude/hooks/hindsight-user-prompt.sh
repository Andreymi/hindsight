#!/opt/homebrew/bin/bash
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

# === LOGGING ===
LOG_FILE="${HOME}/.hindsight/hooks.log"
LOG_LEVEL="${HINDSIGHT_HOOKS_LOG_LEVEL:-INFO}"
HOOK_NAME="user-prompt"

log() {
    local level="$1"
    local message="$2"
    # Уровни: DEBUG=0, INFO=1, WARN=2, ERROR=3
    local -A levels=([DEBUG]=0 [INFO]=1 [WARN]=2 [ERROR]=3)
    local msg_level="${levels[$level]:-1}"
    local cfg_level="${levels[$LOG_LEVEL]:-1}"

    if [[ $msg_level -ge $cfg_level ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] [$HOOK_NAME] $message" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

INPUT=$(cat) || exit 0
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty' 2>/dev/null) || exit 0

# Конфигурация
BANK_ID="${HINDSIGHT_BANK_ID:-hindsight-dev}"
MIN_PROMPT_LENGTH=30
MAX_FACTS=5
CACHE_TTL_SECONDS=45  # Время жизни кеша

# Используем локальный hindsight-embed из проекта
HINDSIGHT_EMBED="${HINDSIGHT_EMBED_PATH:-/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed}"

# Директория кеша и логов
CACHE_DIR="${HOME}/.hindsight/cache"
mkdir -p "$CACHE_DIR" "$(dirname "$LOG_FILE")" 2>/dev/null || true

log "DEBUG" "Started with prompt length=${#PROMPT}"

# Проверяем что hindsight-embed доступен
if [[ ! -x "$HINDSIGHT_EMBED" ]]; then
    log "WARN" "hindsight-embed not found at $HINDSIGHT_EMBED"
    exit 0
fi

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

# === THROTTLING ===
# Вычисляем hash от первых 100 символов для ключа кеша
PROMPT_KEY=$(echo "${PROMPT:0:100}" | md5sum 2>/dev/null | cut -d' ' -f1 || echo "${PROMPT:0:100}" | md5 2>/dev/null)
CACHE_FILE="$CACHE_DIR/prompt-${PROMPT_KEY:0:16}.cache"

# Проверяем кеш
if [[ -f "$CACHE_FILE" ]]; then
    # Проверяем возраст файла
    if [[ "$(uname)" == "Darwin" ]]; then
        FILE_AGE=$(( $(date +%s) - $(stat -f %m "$CACHE_FILE" 2>/dev/null || echo 0) ))
    else
        FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
    fi

    if [[ $FILE_AGE -lt $CACHE_TTL_SECONDS ]]; then
        # Кеш валиден — используем его
        FACTS=$(cat "$CACHE_FILE" 2>/dev/null)
        if [[ -n "$FACTS" ]]; then
            log "INFO" "Cache HIT (age=${FILE_AGE}s)"
            # Формируем компактный контекст
            CONTEXT="## Relevant Hindsight Memory\n"
            while IFS= read -r fact; do
                [[ -n "$fact" ]] && CONTEXT="${CONTEXT}• ${fact}\n"
            done <<< "$FACTS"

            jq -n --arg ctx "$CONTEXT" '{
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": $ctx
                }
            }'
            exit 0
        fi
    fi
fi

log "DEBUG" "Cache MISS, calling recall API"

# Формируем запрос для recall — используем сам prompt как query
# hindsight сам извлечёт ключевые слова через query analyzer
RESULT=$($HINDSIGHT_EMBED memory recall "$BANK_ID" "$PROMPT" \
    -b low --max-tokens 1500 -o json 2>/dev/null) || exit 0

# Извлекаем факты
FACTS=$(echo "$RESULT" | jq -r '.results[]?.text' 2>/dev/null | cut -d'|' -f1 | head -$MAX_FACTS)

# Сохраняем в кеш
echo "$FACTS" > "$CACHE_FILE" 2>/dev/null || true

# Считаем количество фактов
FACTS_COUNT=$(echo "$FACTS" | grep -c . 2>/dev/null || echo 0)
log "INFO" "Recall completed: ${FACTS_COUNT} facts found"

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
