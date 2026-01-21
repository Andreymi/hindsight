# Claude Code Configuration

Конфигурация Claude Code для проекта Hindsight.

## Быстрый старт

```bash
# 1. Скопируйте example файл
cp .claude/settings.local.json.example .claude/settings.local.json

# 2. Замените плейсхолдеры реальными путями
# - ${PROJECT_ROOT} → абсолютный путь к проекту
# - /path/to/hindsight-embed → путь к hindsight-embed CLI
# - your-bank-id → ID вашего memory bank

# 3. Перезапустите Claude Code
```

## Структура

```
.claude/
├── README.md                      # Этот файл
├── settings.local.json            # Ваша локальная конфигурация (в .gitignore)
├── settings.local.json.example    # Шаблон для конфигурации
└── hooks/
    ├── README.md                  # Документация hooks
    ├── hindsight-session-start.sh # SessionStart hook
    ├── hindsight-user-prompt.sh   # UserPromptSubmit hook
    ├── hindsight-smart-stop.py    # Stop hook
    └── hindsight-session-stop.sh  # (deprecated, заменён smart-stop.py)
```

## Hooks

См. [hooks/README.md](hooks/README.md) для полной документации.

| Hook | Событие | Функция |
|------|---------|---------|
| `session-start` | SessionStart | Загрузка контекста проекта |
| `user-prompt` | UserPromptSubmit | Recall по теме промпта |
| `smart-stop` | Stop | Извлечение learnings |

## Требования

- `hindsight-embed` CLI (установлен и доступен)
- `jq` для парсинга JSON
- `python3` для smart-stop.py
