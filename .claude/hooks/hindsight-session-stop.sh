#!/opt/homebrew/bin/bash
# hindsight-session-stop.sh
# Анализирует сессию и сохраняет learnings в Hindsight
#
# НЕ блокирует Claude — только сохраняет в фоне

set -e

# Читаем входные данные
INPUT=$(cat)
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')

# КРИТИЧНО: предотвращаем infinite loop
[[ "$STOP_HOOK_ACTIVE" == "true" ]] && exit 0

# Проверяем наличие транскрипта
[[ -z "$TRANSCRIPT_PATH" || ! -f "$TRANSCRIPT_PATH" ]] && exit 0

# Конфигурация
BANK_ID="${HINDSIGHT_BANK_ID:-hindsight-dev}"
PROJECT_NAME=$(basename "$CWD" 2>/dev/null || echo "unknown")

# Извлекаем последние сообщения assistant (learnings обычно там)
# Берём последние 20 строк транскрипта, фильтруем assistant messages
RECENT_CONTENT=$(tail -20 "$TRANSCRIPT_PATH" 2>/dev/null | \
    jq -r 'select(.role == "assistant") |
           if .content | type == "array" then
               .content[] | select(.type == "text") | .text
           else
               .content // empty
           end' 2>/dev/null | \
    tail -2000)  # Ограничиваем размер

# Если контент слишком короткий — не сохраняем
[[ ${#RECENT_CONTENT} -lt 100 ]] && exit 0

# Сохраняем в фоне (async) чтобы не блокировать завершение
(
    hindsight-embed memory retain "$BANK_ID" \
        "Session summary for project $PROJECT_NAME: $RECENT_CONTENT" \
        --context "session-$PROJECT_NAME" \
        --async \
        -o json 2>/dev/null
) &

exit 0
