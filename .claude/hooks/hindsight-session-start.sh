#!/opt/homebrew/bin/bash
# hindsight-session-start.sh
# Загружает релевантный контекст из Hindsight при старте сессии
#
# Stdout автоматически становится additionalContext для Claude

# === GRACEFUL DEGRADATION ===
# Никогда не блокируем Claude — любая ошибка = тихий exit
trap 'exit 0' ERR SIGTERM SIGINT

# === LOGGING ===
# Используем локальные логи в проекте (CLAUDE_PROJECT_DIR), fallback на глобальные
if [[ -n "$CLAUDE_PROJECT_DIR" ]]; then
    LOG_DIR="$CLAUDE_PROJECT_DIR/.claude/hooks/logs"
else
    LOG_DIR="${HOME}/.hindsight"
fi
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="${HOME}/.hindsight"
LOG_FILE="$LOG_DIR/hooks.log"
LOG_LEVEL="${HINDSIGHT_HOOKS_LOG_LEVEL:-INFO}"
HOOK_NAME="session-start"

log() {
    local level="$1"
    local message="$2"
    local -A levels=([DEBUG]=0 [INFO]=1 [WARN]=2 [ERROR]=3)
    local msg_level="${levels[$level]:-1}"
    local cfg_level="${levels[$LOG_LEVEL]:-1}"

    if [[ $msg_level -ge $cfg_level ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] [$HOOK_NAME] $message" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

# Читаем входные данные
INPUT=$(cat) || exit 0
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null) || exit 0
SOURCE=$(echo "$INPUT" | jq -r '.source // "startup"' 2>/dev/null) || exit 0

log "DEBUG" "Started: source=$SOURCE cwd=$CWD"

# Только при startup или resume
[[ "$SOURCE" != "startup" && "$SOURCE" != "resume" ]] && exit 0

# Конфигурация
BANK_ID="${HINDSIGHT_BANK_ID:-hindsight-dev}"

# Используем локальный hindsight-embed из проекта
HINDSIGHT_EMBED="${HINDSIGHT_EMBED_PATH:-/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed}"

# Проверяем что hindsight-embed доступен
if [[ ! -x "$HINDSIGHT_EMBED" ]]; then
    log "WARN" "hindsight-embed not found at $HINDSIGHT_EMBED"
    exit 0
fi

# Имя проекта из директории
PROJECT_NAME=$(basename "$CWD" 2>/dev/null || echo "unknown")
log "INFO" "Loading context for project: $PROJECT_NAME"

# Recall релевантного контекста
RESULT=$($HINDSIGHT_EMBED memory recall "$BANK_ID" \
    "project $PROJECT_NAME context preferences architecture decisions" \
    -b low --max-tokens 1500 -o json 2>/dev/null) || exit 0

# Извлекаем факты (API возвращает .results[], не .facts[])
# Берём только основную часть до первого " |" для краткости
FACTS=$(echo "$RESULT" | jq -r '.results[]?.text' 2>/dev/null | cut -d'|' -f1 | head -6)

# Если есть факты — выводим как контекст
if [[ -n "$FACTS" ]]; then
    FACTS_COUNT=$(echo "$FACTS" | grep -c . 2>/dev/null || echo 0)
    log "INFO" "Loaded $FACTS_COUNT facts for project $PROJECT_NAME"

    echo "## Hindsight Memory"
    echo ""
    echo "Relevant context for '$PROJECT_NAME':"
    echo "$FACTS" | while IFS= read -r fact; do
        [[ -n "$fact" ]] && echo "- $fact"
    done
else
    log "DEBUG" "No facts found for project $PROJECT_NAME"
fi

exit 0
