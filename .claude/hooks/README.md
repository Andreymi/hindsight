# Hindsight Claude Code Hooks

Claude Code hooks для автоматической интеграции с Hindsight memory system.

## Обзор

| Hook | Событие | Функция |
|------|---------|---------|
| `hindsight-session-start.sh` | `SessionStart` | Загружает контекст проекта при старте сессии |
| `hindsight-user-prompt.sh` | `UserPromptSubmit` | Recall релевантной памяти по теме промпта |
| `hindsight-smart-stop.py` | `Stop` | Извлекает и сохраняет learnings из сессии |

## Установка

### 1. Настройка переменных окружения

Добавьте в `.claude/settings.local.json`:

```json
{
  "env": {
    "HINDSIGHT_EMBED_PATH": "/path/to/hindsight-embed",
    "HINDSIGHT_BANK_ID": "your-bank-id"
  }
}
```

Или используйте defaults:
- `HINDSIGHT_EMBED_PATH`: `/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed`
- `HINDSIGHT_BANK_ID`: `hindsight-dev`

### 2. Подключение hooks в settings.local.json

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/.claude/hooks/hindsight-session-start.sh",
            "timeout": 10
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/.claude/hooks/hindsight-user-prompt.sh",
            "timeout": 8
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/.claude/hooks/hindsight-smart-stop.py",
            "timeout": 35
          }
        ]
      }
    ]
  }
}
```

## Описание hooks

### hindsight-session-start.sh

**Событие:** `SessionStart` (startup, resume)

**Что делает:**
1. Получает имя проекта из `cwd`
2. Делает `recall` контекста проекта (preferences, architecture, decisions)
3. Возвращает релевантные факты как `additionalContext`

**Конфигурация:**
- Timeout: 10 секунд
- Budget: `low` (быстрый recall)
- Max tokens: 1500

### hindsight-user-prompt.sh

**Событие:** `UserPromptSubmit`

**Что делает:**
1. Фильтрует тривиальные промпты (короткие, "ок", "да", etc.)
2. Делает `recall` по теме промпта пользователя
3. Возвращает релевантные факты как `additionalContext`

**Фильтры:**
- Минимум 30 символов
- Игнорирует: `ок`, `да`, `нет`, `спасибо`, `понял`, `1`, `2`, `y`, `n`, etc.

**Throttling:**
- Кеш: `~/.hindsight/cache/prompt-{hash}.cache`
- TTL: 45 секунд
- Повторные запросы используют кеш (~450x быстрее)

**Конфигурация:**
- Timeout: 8 секунд
- Budget: `low`
- Max facts: 5
- Cache TTL: 45 секунд

### hindsight-smart-stop.py

**Событие:** `Stop`

**Что делает:**
1. Извлекает последние 10 сообщений assistant из транскрипта
2. Проверяет наличие "значимого" контента (решения, ошибки, learnings)
3. Использует `reflect` для умного извлечения только важных фактов
4. Сохраняет learnings асинхронно через `retain --async`

**Что сохраняет:**
- Learnings (что узнали нового)
- Decisions (архитектурные решения)
- Failures (что не сработало и почему)
- Procedures (процедуры которые сработали)

**Что НЕ сохраняет:**
- Рутинные операции
- Код (он в git)
- Общие вопросы без выводов
- **Дубликаты** (проверка через Jaccard similarity ≥ 0.35)

**Конфигурация:**
- Timeout: 35 секунд
- Budget: `low`
- Min session turns: 4 (минимум сообщений для анализа)

## Graceful Degradation

Все hooks реализуют graceful degradation:

1. **Проверка бинарника:** `[[ ! -x "$HINDSIGHT_EMBED" ]] && exit 0`
2. **Trap на ошибки:** `trap 'exit 0' ERR SIGTERM SIGINT`
3. **Fallback на каждой команде:** `command || exit 0`

При недоступном `hindsight-embed` hooks тихо выходят с кодом 0, не блокируя Claude.

## Тестирование

```bash
# Тест SessionStart
echo '{"cwd": "/path/to/project", "source": "startup"}' | \
    HINDSIGHT_BANK_ID="your-bank" \
    ./.claude/hooks/hindsight-session-start.sh

# Тест UserPromptSubmit
echo '{"prompt": "как настроить hooks для hindsight"}' | \
    HINDSIGHT_BANK_ID="your-bank" \
    ./.claude/hooks/hindsight-user-prompt.sh

# Тест Stop (нужен транскрипт)
echo '{"transcript_path": "/path/to/transcript.jsonl", "cwd": "/path/to/project"}' | \
    HINDSIGHT_BANK_ID="your-bank" \
    python3 ./.claude/hooks/hindsight-smart-stop.py
```

## Использование в других проектах

Для использования этих hooks в других проектах:

```bash
# Symlink в ~/.claude/hooks (глобально)
ln -s /path/to/patched/.claude/hooks ~/.claude/hooks

# Или скопировать и настроить paths
cp -r /path/to/patched/.claude/hooks /your/project/.claude/
```

## Логгирование

Все hooks пишут логи в `~/.hindsight/hooks.log`.

**Формат:** `YYYY-MM-DD HH:MM:SS [LEVEL] [HOOK] message`

**Уровни:** `DEBUG`, `INFO`, `WARN`, `ERROR`

**Настройка уровня:**
```bash
# В settings.local.json env или в shell
export HINDSIGHT_HOOKS_LOG_LEVEL=DEBUG  # показывать все логи
export HINDSIGHT_HOOKS_LOG_LEVEL=INFO   # default
export HINDSIGHT_HOOKS_LOG_LEVEL=ERROR  # только ошибки
```

**Просмотр логов:**
```bash
# Последние записи
tail -20 ~/.hindsight/hooks.log

# Следить в реальном времени
tail -f ~/.hindsight/hooks.log

# Только ошибки
grep ERROR ~/.hindsight/hooks.log
```

**Пример вывода:**
```
2026-01-21 22:22:02 [INFO] [user-prompt] Recall completed: 5 facts found
2026-01-21 22:22:15 [INFO] [session-start] Loaded 6 facts for project patched
2026-01-21 22:25:30 [INFO] [smart-stop] Retained learnings for project patched (512 chars)
```

## Зависимости

- **Bash 5+** — для ассоциативных массивов (macOS: `brew install bash`)
- `jq` — для парсинга JSON в bash
- `python3` — для smart-stop.py
- `hindsight-embed` — CLI для Hindsight

### Установка Bash 5 на macOS

```bash
brew install bash

# Проверка версии
/opt/homebrew/bin/bash --version
# GNU bash, version 5.3.9(1)-release

# Скрипты используют shebang: #!/opt/homebrew/bin/bash
# Системный bash остаётся /bin/bash (3.2)
```
