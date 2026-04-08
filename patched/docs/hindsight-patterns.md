# Hindsight — Справочник паттернов (v0.5.0)

Проверенные паттерны использования Hindsight для AI-агентов с долгосрочной памятью.
Актуально для v0.5.0 (апрель 2026). API: `hindsight-api-slim/`, CLI: `hindsight-embed` + `hindsight` (Rust).

---

## 1. Entity Labels v2 — таксономическая классификация фактов

### Что это
Контролируемый словарь `key:value`, настраиваемый per-bank. При retain LLM классифицирует каждый извлечённый факт по заданной таксономии. Labels сохраняются как entity-связи в графе и участвуют в graph retrieval.

### Изменения в v0.4.15

| Было (v0.4.14) | Стало (v0.4.15) |
|----------------|-----------------|
| `free_values: bool` + `multi_value: bool` | `type: "value" \| "multi-values" \| "text"` |
| Нет auto-tagging | `tag: true` — label-entity автоматически добавляется в `tags[]` факта |
| Entity resolution: full scan | `retain_entity_lookup: "trigram"` (default) — pg_trgm GIN index |
| `retain_free_form_entities` | Переименовано в `entities_allow_free_form` |
| Flat list в config API | Backward-compatible: `_migrate_label_group()` конвертирует старый формат |
| Нет BM25 по entity names | `text_signals` колонка — entity names в BM25 index без загрязнения текста факта |

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Labels не ретроспективны | Применяются только к новым retain. Для существующих данных: удалить документ → re-retain |
| 2 | **`tag: true` — killer feature** | Labels с `tag: true` автоматически попадают в `tags[]` → recall с `tags: ["domain:business"]` фильтрует по домену. **Проверено: работает!** |
| 3 | `optional: false` гарантирует | LLM **обязан** заполнить label (structured output enforcement). priority:high/medium на каждом факте |
| 4 | `type: "text"` — свободный ввод | LLM сам придумывает значение. Пример: `source_type:team notes` из context |
| 5 | Graph кластеризация | Факты с одинаковым label связаны через общую entity, graph traversal находит тематические кластеры |
| 6 | Сосуществование с entities | `entities_allow_free_form: true` (default) — labels работают рядом с обычными named entities |

### Типы LabelGroup (v0.4.15 синтаксис)

| `type` | Описание | Structured Output | Пример |
|--------|----------|-------------------|--------|
| `"value"` | Enum — одно значение из списка `values` | `Literal["v1","v2"]` или `Literal[...] \| None` | priority: critical/high/medium/low |
| `"multi-values"` | Несколько значений из списка `values` | `list[Literal["v1","v2"]]` | domain: [engineering, product] |
| `"text"` | Свободный текст от LLM | `str \| None` (всегда optional) | source_type: "team meeting notes" |

### Настройка

```bash
# Включить labels для банка (через API, нет CLI-команды для entity_labels)
curl -X PATCH "http://localhost:8888/v1/default/banks/{bank_id}/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"entity_labels": [...], "entities_allow_free_form": true}}'
```

### Пример таксономии (v0.4.15)

```json
{
  "updates": {
    "entity_labels": [
      {
        "key": "priority",
        "description": "Importance level",
        "type": "value",
        "optional": false,
        "tag": true,
        "values": [
          {"value": "critical", "description": "Must not forget"},
          {"value": "high", "description": "Important"},
          {"value": "medium", "description": "Useful to know"},
          {"value": "low", "description": "Nice to have"}
        ]
      },
      {
        "key": "domain",
        "description": "Knowledge domain",
        "type": "multi-values",
        "optional": true,
        "tag": true,
        "values": [
          {"value": "engineering"},
          {"value": "product"},
          {"value": "business"},
          {"value": "personal"}
        ]
      },
      {
        "key": "source_type",
        "description": "How this information was obtained",
        "type": "text",
        "optional": true,
        "tag": false
      }
    ],
    "entities_allow_free_form": true
  }
}
```

### Фильтрация recall по тегам (NEW в v0.4.15!)

```bash
# Через API — recall только бизнес-факты
curl -X POST "http://localhost:8888/v1/default/banks/{bank_id}/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "what do we know?", "tags": ["domain:business"], "tags_match": "any"}'

# Результат: только факты с tag domain:business
# tags_match: "any" (любой из тегов), "all" (все теги), "any_strict", "all_strict"
```

### Известные ограничения
- Параллельный retain в одну секунду может получить одинаковый `document_id` (timestamp-based) — использовать `--doc-id` или разносить по времени
- ~~CLI `recall` не поддерживает `--tag` — только через API~~ **Исправлено**: CLI поддерживает `--tags`, `--tags-match`, `--tag-groups` (commit `02150a42`, `752c62c8`)
- CLI `set-config` не поддерживает entity_labels — только через curl PATCH

---

## 2. Mental Models — консолидированные знания

### Что это
Mental model = сохранённый результат reflect по `source_query`. При последующих reflect агент сначала ищет в mental models, потом дополняет recall. Auto-refresh пересчитывает модель после каждой consolidation.

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Blind spot при refresh | Если mental model уже содержит достаточно контента (12-21K chars), агент НЕ вызывает recall и новые факты не попадают. Root cause: refresh использует `budget=low` |
| 2 | Конкретные запросы работают | `"Расскажи про Go SDK"` → reflect находит через recall. Обобщённые запросы → полагается на mental model |
| 3 | Качество зависит от source_query | Чем шире и конкретнее запрос, тем полнее модель. Формулировка: "Перечисли ВСЕ X, включая недавние изменения" |
| 4 | Reflect ≠ кеш | Reflect всегда перегенерирует ответ, mental model — один из источников, не единственный |
| 5 | Галлюцинации при неточном запросе | Неточная формулировка source_query → LLM интерпретирует термины буквально (Rust CLI как "клиент") |

### Команды

```bash
hindsight-embed mental-model create <bank> "<name>" "<source_query>"
hindsight-embed mental-model list <bank>
hindsight-embed mental-model get <bank> <id>
hindsight-embed mental-model refresh <bank> <id>
```

### Рекомендации
- Используй mental models для **FAQ и status reports** — предсказуемые запросы, стабильные ответы
- **Не полагайся на auto-refresh для полноты** — ручной review после major data changes
- **Множественные узкие модели > одна большая** — 5 узких моделей лучше чем 1 "расскажи всё"
- Для **динамичных данных** — prefer прямой reflect без mental models (свежее, но медленнее)

### Upstream improvement candidates
- Engine `reflect_async()` поддерживает `budget`, но Refresh API endpoint (`PUT /mental-models/{id}`) **не экспонирует** этот параметр — refresh всегда использует default budget. Нужен budget param в REST API
- `exclude_mental_model_ids` работает при refresh — модель не видит саму себя, blind spot из-за shallow search

### UPDATE v0.4.22: Blind spot частично решён
Паттерн #1 (blind spot при refresh) теперь адресуем через `trigger.tags_match` и `trigger.tag_groups` в конфигурации mental model. Refresh по умолчанию использует `all_strict` → untagged memories невидимы. Установив `trigger: { tags_match: "any" }`, можно включить untagged content в refresh scope. **Подробности → Раздел 12.**

---

## 3. Shared Banks — мульти-агентные паттерны

### Что это
Архитектурный паттерн: shared bank (общая база знаний) + personal banks (личная память каждого агента/пользователя). Банки строго изолированы, мульти-банковый recall — на стороне клиента.

### Паттерн оркестрации

```
shared bank   = "что знает команда"  (пишут все, читают все)
personal bank = "что знаю я"         (пишет один, читает один)
client        = решает, какие банки опрашивать для конкретного запроса
```

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Write-to-shared | Несколько агентов пишут в shared bank → reflect синтезирует целостную картину |
| 2 | Изоляция личных банков | Personal bank не видит знания других агентов. Для полной картины — multi-bank recall |
| 3 | Коллективный entity graph | Entity graph в shared bank строится из записей всех агентов. Автоматическая карта знаний проекта |
| 4 | Emergent knowledge layering | Знания стратифицируются по ролям: shared="что нужно", researcher="почему", developer="как" |

### Пример архитектуры (dev team)

```
shared-project (shared)       agent-researcher (personal)   agent-developer (personal)
├── Требования                ├── Анализ технологий          ├── Прогресс реализации
├── Архитектурные решения     ├── Обоснования выбора         ├── Техдолг
├── Status updates            ├── Сравнения альтернатив      ├── Найденные проблемы
├── ALERT-ы (от researcher)   └── Детали исследований        └── Code review заметки
└── RESOLVED (от developer)
```

### Известные проблемы
- **Entity дедупликация**: `Omega` и `project Omega` — две разные сущности. Нужен entity resolution / алиасы
- **Multi-bank recall**: нет серверного API для запроса нескольких банков одновременно

---

## 4. Cross-Agent Knowledge Promotion — обмен знаниями между агентами

### Что это
Паттерн, при котором агент находит важную информацию в своём личном банке и "промоутит" её в shared bank, делая доступной для всей команды.

### Workflow

```
1. Агент пишет находку в свой личный банк (raw notes)
2. Reflect на личном банке → triage: critical vs nice-to-have
3. Критичные находки → retain в shared bank с маркировкой источника
4. Другие агенты → recall/reflect из shared bank → видят промоутнутые знания
5. При решении проблемы → RESOLVED запись в shared bank (closure)
```

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Reflect как triage-фильтр | Перед promotion — reflect для классификации важности. Предотвращает information overload |
| 2 | Атрибуция через текст | Пишем "ALERT от agent-X: ..." → Hindsight извлекает `Involving: agent-X` автоматически |
| 3 | Alert→Fix lifecycle | ALERT→RESOLVED в shared bank. Reflect реконструирует полный цикл: кто нашёл → кто исправил |
| 4 | Двунаправленный поток | researcher→shared (находки) + developer→shared (решения). Shared bank = журнал решений |

### Форматы promotion-записей

```
# Alert
"SECURITY ALERT от [agent]: [описание]. ДЕЙСТВИЕ: [что делать]. Приоритет: CRITICAL."

# Resolution
"RESOLVED от [agent]: [что исправлено]. Commit: [hash]. [подтверждение тестов]."

# Insight (sales)
"INSIGHT от [seller]: [паттерн/приём]. Подтверждено сделкой [клиент] ([план], $[сумма]/мес)."
```

---

## 5. Reflect как AI-коуч (Sales Team Scenario)

### Что это
Применение shared bank + promotion для команды продаж: shared playbook с продуктом/конкурентами/возражениями + личные банки продавцов. Reflect на shared bank генерирует персонализированный коучинг.

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Reflect как AI-коуч | Новичок спрашивает "как работать с возражением X?" → reflect генерирует полный скрипт встречи |
| 2 | Масштабируемый коучинг | Знания лучших продавцов автоматически доступны новичкам через promotion + reflect |
| 3 | INSIGHT-формат | Маркировка источника + доказательная база (подтверждено сделкой) сохраняется в атрибуции |

### Пример архитектуры (sales team)

```
sales-playbook (shared)     seller-alice (personal)     seller-boris (personal)
├── Продукт и тарифы        ├── Клиент: FinTechCorp     ├── Клиент: StartupHub
├── Конкурентный анализ      ├── Клиент: MedTech         ├── Клиент: LogiTrack
├── Работа с возражениями    ├── Личные заметки          ├── Вопросы, затруднения
├── INSIGHT-ы от Alice       └── Уроки из сделок         └── Первый опыт
└── INSIGHT-ы от Boris
```

---

## 6. Knowledge Conflicts — разрешение противоречий

### Что это
Когда несколько агентов/пользователей записывают противоречивые факты в shared bank, reflect использует разные стратегии разрешения в зависимости от типа конфликта.

### Стратегии разрешения

| Тип конфликта | Пример | Стратегия reflect | Качество |
|--------------|--------|-------------------|----------|
| Факты из разных контекстов | "Пилот = 3 мес" vs "Пилот = 2 нед" | **Контекстная сегментация** — "оба правы: Enterprise 3 мес, Startup 2 нед" | Отлично |
| Разные стратегии | "Сразу Business" vs "Начни со Starter" | **Trade-off анализ** — оба варианта с условиями и атрибуцией | Хорошо |
| Субъективные оценки | "У Spacelift лучше UI" vs "Spacelift — слабый" | **Bias к позитиву** — подавляет minority opinion | Проблема |

### Disposition:skepticism как решение

| skepticism | Поведение при конфликтах |
|-----------|-------------------------|
| 1-3 (default) | Склоняется к позитивному/мажоритарному мнению, подавляет minority |
| 4-5 | Сохраняет minority opinion, признаёт риски, более сбалансированный синтез |

**Рекомендация**: для shared bank с конфликтующими мнениями ставить **skepticism=4-5**:

```bash
hindsight-embed bank update <bank_id> --skepticism 5
```

---

## 7. Retain & Extract — режимы и стратегии

### PydanticAI Integration (`hindsight-pydantic-ai`)

Полноценный Python-пакет для PydanticAI агентов. Async-native, три tool-а + auto-inject воспоминаний.

```python
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)
```

| Параметр | Описание |
|----------|----------|
| `budget` | low/mid/high — глубина recall |
| `include_retain/recall/reflect` | Какие tools давать агенту |
| `tags`, `recall_tags`, `recall_tags_match` | Фильтрация recall |
| `memory_instructions(query=...)` | Кастомный recall-запрос для instructions injection |

**Статус**: пакет в репо, требует `pip install hindsight-pydantic-ai`. Не тестировали локально.

### `timestamp="unset"` — контент без даты

Для художественных текстов, справочников, статических знаний — `event_date = NULL`.

```python
client.retain(bank_id="ref", content="E=mc²", timestamp="unset")
```

**Статус**: ✅ Исправлено в patched (commit `eb0ee5bd`, 2026-03-05). Guard `if event_date is not None:` в `fact_extraction.py` перед `.isoformat()`. Миграция `aa2b3c4d5e6f` применена, `event_date` nullable в БД.

### Recall Performance — −50% latency

- Single-query chunk fetch вместо batched while-loop
- 3 partial indexes на temporal fields (CONCURRENTLY)
- pg_trgm GIN index на `entities.canonical_name`
- 2 covering indexes на `memory_links`

На банке 97K units: **p50 1.2s → 0.6s, mean 1.5s → 0.8s**

### Reflect — не падает на больших банках

Proactive token-budget guard: считает токены перед каждым tool call. При превышении `reflect_max_context_tokens` (default 100K) — force-synthesis.

### Bank-scoped validation

`validate_bank_read/write` hooks + `allowed_bank_ids` в RequestContext. Для SaaS: API key → доступ только к своим банкам.

### Gemini/Vertex AI Safety Settings

Per-bank конфигурация safety settings. 6 threshold levels, UI в control plane.

### List Documents — tags filtering

`GET /banks/{bank_id}/documents?tags=x&tags_match=any` — фильтрация документов по тегам.

### Tag Groups — составные boolean-фильтры (v0.4.18)

Простые `tags` + `tags_match` поддерживают один предикат. `tag_groups` позволяют строить **сложные вложенные boolean-выражения** (AND/OR/NOT) для фильтрации.

```bash
# CLI — compound filter: (user:alice AND step:5) OR priority:high
hindsight-embed memory recall <bank> "query" \
  --tag-groups '[{"tags":["user:alice","step:5"],"match":"all_strict"},{"or":[{"tags":["priority:high"],"match":"any_strict"}]}]'

# API — тот же запрос
curl -X POST ".../memories/recall" -d '{
  "query": "...",
  "tag_groups": [
    { "tags": ["user:alice", "step:5"], "match": "all_strict" },
    { "or": [{ "tags": ["priority:high"], "match": "any_strict" }] }
  ]
}'
```

| Оператор | Описание | Пример |
|----------|----------|--------|
| `tags` + `match` (top-level) | AND между группами | `[{tags:["a"]}, {tags:["b"]}]` = a AND b |
| `or` | OR между вложенными группами | `{"or": [{tags:["a"]}, {tags:["b"]}]}` = a OR b |
| `not` | Исключение | `{"not": {tags:["draft"], match:"any_strict"}}` |

**Когда использовать**: multi-user сценарии ("факты Alice или Bob, но не draft"), сложные pipeline с многоуровневыми тегами.

### Observation Scopes — гранулярная консолидация (v0.4.18)

Контролирует, как создаются observations (consolidated summaries) относительно тегов. Настраивается per-retain.

| Scope | Описание | Use case |
|-------|----------|----------|
| `combined` (default) | Один observation на весь документ | Одноавторский контент |
| `per_tag` | Отдельный observation для каждого тега | "Что Alice знает через все сессии?" |
| `all_combinations` | Observation для каждой комбинации тегов | Детальная аналитика |
| `custom` | Явно указанные подмножества тегов | Гибридные сценарии |

```bash
# Per-tag observations — каждый пользователь получает свой observation
curl -X POST ".../memories/retain" -d '{
  "content": "...",
  "tags": ["user:alice", "session:3"],
  "observation_scope": "per_tag"
}'
```

### Retain Strategies — именованные конфигурации (v0.4.19)

Стратегии позволяют определить per-bank **именованные наборы настроек retain** и выбирать их при каждом вызове через `--strategy`.

```bash
# Определить стратегии для банка
curl -X PATCH ".../banks/{bank_id}/config" -d '{
  "updates": {
    "retain_strategies": {
      "documents": {
        "retain_extraction_mode": "chunks",
        "retain_chunk_size": 800
      },
      "conversations": {
        "retain_extraction_mode": "concise",
        "retain_chunk_size": 3000
      },
      "raw": {
        "retain_extraction_mode": "verbatim"
      }
    }
  }
}'

# Использовать стратегию при retain
hindsight-embed memory retain <bank> "content" --strategy documents
hindsight-embed memory retain <bank> "chat log" --strategy conversations
```

### Extraction Modes: `verbatim` и `chunks` (v0.4.19)

Два новых режима извлечения фактов, дополняющие стандартные `concise`/`verbose`:

| Mode | LLM calls | Описание | Use case |
|------|-----------|----------|----------|
| `concise` | ✅ Да | LLM извлекает ключевые факты (default) | Разговоры, заметки |
| `verbose` | ✅ Да | LLM извлекает все детали | Документация, техспеки |
| `verbatim` | ✅ Минимально | Чанки as-is + LLM для metadata | RAG-подобный, быстрый |
| `chunks` | ❌ Нет | Чанки as-is, zero LLM | Максимальная скорость, структурированные данные |

```bash
# Verbatim — сохраняет текст, LLM добавляет metadata
hindsight-embed memory retain <bank> - --strategy raw < document.txt

# Chunks — полностью без LLM
curl -X POST ".../memories/retain" -d '{
  "content": "...",
  "retain_extraction_mode": "chunks"
}'
```

### Reflect — structured output (v0.4.18)

Reflect поддерживает `response_schema` для получения структурированного JSON-ответа:

```bash
curl -X POST ".../banks/{bank_id}/reflect" -d '{
  "query": "Оцени risk score проекта",
  "response_schema": {
    "type": "object",
    "properties": {
      "risk_score": { "type": "number" },
      "risks": { "type": "array", "items": { "type": "string" } },
      "recommendation": { "type": "string" }
    },
    "required": ["risk_score", "risks", "recommendation"]
  }
}'
```

---

## 8. Tags + Context + Mental Models — Управление фокусом reflect

### Четыре инструмента фокусировки

| Инструмент | Что делает | Когда использовать |
|---|---|---|
| `--tags` + `any` (default) | **Boost** тегированных моделей/фактов, нетегированные по-прежнему видны | Мягкая фокусировка на домене |
| `--tags` + `all_strict` | **Жёсткий фильтр** — только memories с указанными тегами | Multi-tenant изоляция, security scoping |
| `--tag-groups` (v0.4.18) | **Compound boolean** — AND/OR/NOT комбинации тегов | Сложные multi-user / multi-domain запросы |
| `--context` | Инъекция текущей ситуации в system prompt reflect-агента | Всегда, когда у вызывающего есть контекст, которого нет в банке |

### Tags на Mental Models — паттерны

```
ПРАВИЛЬНО:  мульти-теги + any match
            implementation-gaps → type:adr, type:phase, type:data
            reflect --tags type:adr → boost adr-моделей, остальные тоже видны

НЕПРАВИЛЬНО: узкие теги + all_strict
             implementation-gaps → type:adr (только)
             reflect --tags type:adr --tags-match all_strict → потеря кросс-доменных моделей
```

**Ключевой принцип**: mental models — это *синтез*, они агрегируют знания из нескольких доменов. Узкий тег на синтетической модели = ложная классификация. Мульти-теги отражают все домены модели.

### `--context` — оперативная память reflect-агента

`context` вставляется в system prompt как `## Additional Context` и виден на **каждом шаге** agentic loop (поиск + синтез).

```bash
# Без context — ответ "в вакууме"
hindsight-embed memory reflect bank "Какие риски при переходе на Phase 7?"

# С context — ответ учитывает текущую ситуацию
hindsight-embed memory reflect bank "Какие риски при переходе на Phase 7?" \
  --context "Мы на Phase 4 (Contacts). Беспокоят breaking changes в schema."
```

**Что передавать в context**:
- Текущая задача/фаза, которая ещё не retain'd
- Роль спрашивающего ("Я frontend-разработчик, интересует UI")
- Ограничения ("бюджет 2 дня", "нельзя менять API контракт")
- Свежие решения, принятые в текущей сессии

**Tags vs Context vs Tag Groups — ортогональны**:
- Tags = scoping *what to search* (простой фильтр/boost по memories)
- Tag Groups = compound scoping *what to search* (boolean AND/OR/NOT комбинации)
- Context = framing *how to reason* (как интерпретировать найденное)
- Комбинация: `--tags type:security --context "Готовим SOC2 аудит, фокус на data flow"`
- Сложная комбинация: `--tag-groups '[{"tags":["user:alice"],"match":"all_strict"},{"or":[{"tags":["step:5"],"match":"any_strict"}]}]' --context "Ревью Phase 5"`

### Tags matching modes — шпаргалка

| Mode | Untagged memories | Тегированные memories | Use case |
|------|---|---|---|
| `any` (default) | **Включены** | Включены если хотя бы 1 тег совпал | Аналитика, boost |
| `all` | **Включены** | Включены если все указанные теги совпали | Точный boost |
| `any_strict` | **Исключены** | Включены если хотя бы 1 тег совпал | Изоляция по домену |
| `all_strict` | **Исключены** | Включены если все указанные теги совпали | Multi-tenant security |

### Паттерн: LLM-агент работает с Hindsight

Типичный workflow для AI-агента, интегрированного с Hindsight:

```
1. RECALL  — "что я знаю об этом?" (семантический поиск, быстрый)
   hindsight-embed memory recall bank "тема" --tags domain:x
   # или compound: --tag-groups '[{"tags":["user:X","domain:Y"],"match":"all_strict"}]'

2. REFLECT — "что я думаю об этом?" (agentic reasoning, глубокий)
   hindsight-embed memory reflect bank "вопрос" \
     --context "текущая задача" --tags domain:x -b mid

3. RETAIN  — "запомню это на будущее" (после выполнения задачи)
   hindsight-embed memory retain bank "что узнал/сделал" --async
   # для документов: --strategy documents (verbatim/chunks mode, быстрее)

4. Периодически: CONSOLIDATION (автоматическая) + MENTAL MODEL REFRESH
```

**Антипаттерны LLM-агента**:
- Reflect без context → ответ "в вакууме", не учитывает текущую задачу
- `all_strict` на аналитическом банке → потеря кросс-доменных знаний
- Retain без `--async` в середине работы → блокировка на 5-15 сек
- Один широкий reflect вместо recall+reflect → перерасход токенов
- Забыть retain после выполнения задачи → знания теряются

---

## 9. Production Best Practices (из official docs)

### Bank Missions — настройка качества извлечения

Три mission'а определяют поведение банка. **Misconfigured missions = самая частая причина плохого качества памяти.**

| Mission | Куда инъектируется | Пример |
|---------|-------------------|--------|
| `retain_mission` | Prompt факт-экстракции при retain | `"Extract technical decisions, API choices, blockers. Ignore greetings."` |
| `observations_mission` | Консолидация (observation synthesis) | `"Identify durable patterns, contradictions. Ignore transient states."` |
| `reflect_mission` | System prompt reflect-агента | `"You are a senior developer. Be direct and opinionated."` |

```bash
# Настроить missions через API
curl -X PATCH ".../banks/{bank_id}/config" -d '{
  "updates": {
    "retain_mission": "Extract architectural decisions, blockers, dependencies. Ignore small talk.",
    "observations_mission": "Track evolving preferences, recurring patterns, contradictions.",
    "reflect_mission": "You are a technical advisor. Reference past decisions. Be concise."
  }
}'
```

### Content Format — что передавать в retain

| Формат | Рекомендация |
|--------|-------------|
| JSON conversation array | **Preferred** — сохраняет structure, roles, timestamps |
| `[timestamp] role: text` | Приемлемо для plain text |
| Markdown / HTML / raw text | Для документов и заметок |
| Pre-summarized | **Избегать** — теряет entity relationships, temporal markers |

**Поле `context` в retain** — описывает *природу контента*. Сильно влияет на качество экстракции:

```bash
# Хорошо — конкретно
hindsight-embed memory retain bank "content" --context "Architecture review session for payments service"

# Плохо — бесполезно
hindsight-embed memory retain bank "content" --context "some data"
```

### Disposition Profiles — рекомендуемые предустановки

| Тип агента | skepticism | literalism | empathy |
|-----------|-----------|-----------|---------|
| Code review | 4 | 5 | 1 |
| Customer support | 2 | 3 | 4 |
| Personal assistant | 2 | 2 | 4 |
| Medical assistant | 5 | 4 | 3 |
| Research assistant | 4 | 4 | 2 |

### Tag Naming Conventions

| Pattern | Пример | Use case |
|---------|--------|----------|
| `user:<id>` | `user:alice` | Per-user изоляция |
| `session:<id>` | `session:s_abc` | Session-scoped memories |
| `team:<name>` | `team:engineering` | Shared team knowledge |
| `topic:<name>` | `topic:billing` | Domain filtering |
| `scope:<name>` | `scope:private` | Visibility tiers |

**Multi-tenant minimum**: каждый retain пользовательских данных **обязан** включать `user:<id>`. Без тега — memory глобально видна.

### Anti-patterns (расширенный список)

| Антипаттерн | Проблема | Исправление |
|------------|---------|-------------|
| Pre-summarize перед retain | Теряет entities, temporal markers | Retain raw content |
| Random UUID как `document_id` | Дубликаты документов | Stable session/ticket IDs |
| Пропускать `context` поле | Низкое качество экстракции | Всегда описывать тип данных |
| `metadata` для фильтрации | Metadata не фильтруемо | Использовать `tags` |
| Vague missions | Шумные, low-value memories | Конкретные домены + что игнорировать |
| `any` match для multi-tenant | Утечка между пользователями | `any_strict` / `all_strict` |
| Retain + recall в одном запросе | Retained ещё не проиндексированы | Retain в конце turn, recall в начале следующего |
| Одна mental model на всё | Низкая accuracy, долгий refresh | Одна модель на knowledge dimension |
| `high` budget на каждый recall | Дорого, медленно, обычно не нужно | `low` для lookups, `mid` default |

---

## 10. Performance & Cost Optimization (из official docs)

### Latency Reference

| Операция | Budget | Латентность | Bottleneck |
|----------|--------|------------|-----------|
| Recall | low | 50-100ms | — |
| Recall | mid | 100-300ms | Re-ranker (CPU) |
| Recall | high | 300-500ms | Re-ranker (CPU) |
| Reflect | — | 600-2600ms | LLM generation |
| Retain | — | 500-2000ms/batch | LLM fact extraction |
| Vector search | — | 10-50ms | DB (HNSW) on 100K+ facts |

### Рекомендуемые LLM модели

Hindsight **не требует frontier models** — факт-экстракция хорошо структурирована:
- **Retain/Consolidation**: `gpt-oss-20b` via Groq (быстро, дёшево)
- **Reflect**: `gpt-oss-120b` via Groq (надёжный tool calling)
- **Self-hosted**: vLLM, TGI на GPU кластерах

### Cost Optimization

| Приём | Экономия | Как включить |
|-------|---------|-------------|
| Batch API | **−50% LLM cost** | `HINDSIGHT_API_RETAIN_BATCH_ENABLED=true` (OpenAI, Groq). Результаты в течение 24ч |
| Async retain | Фоновая обработка | `--async` flag, разгружает user-facing path |
| Budget подбор | Меньше reranker cycles | `low` для simple lookups, `mid` default |
| Chunk size 1000-2000 tokens | Меньше batches | Настройка через retain strategies |
| Auto batch splitting | Оптимальные sub-batches | Автоматически при async retain > 10K tokens |

### Monitoring

Prometheus метрики на `/metrics`:
- `hindsight_recall_duration_seconds` — latency recall
- `hindsight_reflect_duration_seconds` — latency reflect
- `hindsight_retain_items_total` — throughput retain

### Scaling

- **Горизонтальное**: несколько API instances за load balancer + shared PostgreSQL
- **Concurrency**: 100+ simultaneous requests
- **LLM rate limits**: распределить нагрузку по нескольким API keys (60-500 RPM per key)

---

## 11. Delta Retain — оптимизация повторного retain

### Что это
При retain с тем же `document_id` (upsert) Hindsight вычисляет SHA256-хэш каждого чанка и сравнивает с существующими. Неизменённые чанки **пропускают LLM-экстракцию** — обрабатываются только новые/изменённые.

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Upsert = delta | Повторный `retain(content, document_id="X")` обрабатывает только diff. Экономия 80-95% LLM-вызовов при инкрементальных обновлениях |
| 2 | Удалённые чанки каскадно чистятся | Чанки → memory_units → links — всё cascade-deleted при исчезновении чанка из нового контента |
| 3 | Tags обновляются без re-extraction | Изменение тегов при upsert применяется ко всем чанкам, включая неизменённые |
| 4 | `document_metadata` обновляется всегда | Metadata обновляется независимо от изменений в контенте |

### Use case для sup
Периодический re-import заметок контактов, логов активности, документов — при каждом retain обрабатывается только дельта. Особенно полезно для real-time чатов, где `document_id` = session ID и контент растёт с каждым сообщением.

### Пример

```bash
# Первый retain — полная обработка (100 чанков → 100 LLM-вызовов)
hindsight-embed memory retain bank "full document" --doc-id "contact-123-notes"

# Повторный retain — delta (добавлен 1 абзац → 1 LLM-вызов, 99 пропущены)
hindsight-embed memory retain bank "full document + new paragraph" --doc-id "contact-123-notes"
```

---

## 12. Mental Model Tag Filtering — управление scope при refresh

### Что это
Поля `trigger.tags_match` и `trigger.tag_groups` в конфигурации mental model. Контролируют, какие memories попадают в scope при auto-refresh.

### Решённая проблема
Раздел 2, паттерн #1 (blind spot): refresh по умолчанию использует `all_strict` matching → untagged memories невидимы → модель не обновляется новыми данными. Теперь можно явно указать `tags_match: "any"` в trigger.

### Ключевые паттерны

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | `tags_match: "any"` в trigger | Модель видит и тегированные, и untagged memories при refresh. **Решает blind spot** |
| 2 | `tag_groups` в trigger | Compound boolean filtering при refresh — `(type:adr AND domain:auth) OR priority:critical` |
| 3 | Backward-compatible | Без `trigger.tags_match` поведение не меняется (default = `all_strict`) |
| 4 | Комбинация с `auto_refresh` | `{ tags_match: "any", refresh_after_consolidation: true }` — auto-refresh с широким scope |

### API пример

```bash
# Обновить trigger mental model — расширить scope refresh
curl -X PATCH ".../banks/{bank_id}/mental-models/{id}" -d '{
  "trigger": {
    "tags_match": "any",
    "refresh_after_consolidation": true
  }
}'
```

### Рекомендация
Для mental models на банках с тегированными данными (multi-user, multi-domain) — всегда устанавливать `trigger.tags_match: "any"`. Иначе untagged memories (часто = самые ранние) будут невидимы.

---

## 13. Infrastructure Features

Фичи, которые не требуют отдельных разделов, но важны для production.

| Фича | Версия | Описание | Конфигурация |
|------|--------|----------|-------------|
| Reflect wall-clock timeout | v0.4.20 | Жёсткий таймаут agentic loop (default 300s). При превышении — partial answer | `HINDSIGHT_API_REFLECT_WALL_TIMEOUT=300` |
| `fact_types` в reflect | v0.4.20 | Ограничить reflect определёнными типами фактов | `fact_types: ["world"]` в запросе |
| `exclude_mental_models` в reflect | v0.4.20 | Reflect игнорирует mental models, использует только raw facts | `exclude_mental_models: true` в запросе |
| Audit Logging | v0.4.21 | Fire-and-forget логирование операций (action, transport, duration, metadata) | `HINDSIGHT_API_AUDIT_LOG_ENABLED=true` |
| LiteLLM provider | v0.4.21 | 100+ LLM-бэкендов (Bedrock, Azure, Together, Fireworks...) через litellm SDK | `HINDSIGHT_API_LLM_PROVIDER=litellm` |
| `max_observations_per_scope` | v0.4.21 | Лимит observations per tag scope при consolidation. Предотвращает взрыв на широких банках | Per-bank config |
| Document Metadata API | v0.4.22 | `document_metadata` field в `GET .../documents` и `GET .../documents/{id}` | Устанавливается через `metadata` при retain |
| `HINDSIGHT_API_LLM_EXTRA_BODY` | v0.4.22 | JSON dict, merged в `extra_body` LLM-запроса. Для provider-specific params | Env var |
| Experience classification fix | v0.4.22 | Текст от первого лица агента ("I changed X") → `experience` fact_type вместо `world` | Автоматически |
| MCP stateless HTTP | v0.4.21 | POST-only MCP mode без SSE/GET. Для Claude Code и serverless | `HINDSIGHT_API_MCP_STATELESS=true` |
| MCP `filter_mcp_tools` hook | v0.4.21 | Extension hook для per-user фильтрации видимых MCP-инструментов | Extension API |
| MCP `strategy` param | v0.4.21 | Retain MCP tool принимает `strategy` для named retain configs | В MCP tool call |
| `X-Ignored-Params` header | v0.4.22 | API предупреждает о неизвестных параметрах в ответном header | Автоматически |

### Обновление рекомендуемых LLM моделей

С v0.4.21 появился **LiteLLM provider** (`HINDSIGHT_API_LLM_PROVIDER=litellm`), открывающий доступ к 100+ бэкендам:

| Сценарий | Рекомендация | Провайдер |
|----------|-------------|-----------|
| Retain/Consolidation | Быстрая, дешёвая модель | Groq (`gpt-oss-20b`), LiteLLM (`bedrock/...`), Ollama |
| Reflect | Надёжный tool calling | Groq (`gpt-oss-120b`), LiteLLM (`azure/gpt-4o`) |
| Zero-LLM (chunks mode) | Без модели | `HINDSIGHT_API_LLM_PROVIDER=none` (v0.4.21) |
| Self-hosted | vLLM, TGI | `HINDSIGHT_API_LLM_EXTRA_BODY` для custom params |

---

## Сводная таблица фич

| Фича | Версия | Статус | Ключевой вывод |
|------|--------|--------|----------------|
| Entity Labels v2 | v0.4.15 | ✅ Работает | `tag: true` + `optional: false` — auto-tagging + обязательная классификация |
| Mental Models | v0.4.14 | ✅ С оговоркой | Blind spot при refresh (API не экспонирует budget); узкие модели лучше широких |
| Shared Banks | v0.4.14 | ✅ Работает | Клиентская оркестрация multi-bank recall; entity dedup — проблема |
| Knowledge Promotion | v0.4.14 | ✅ Работает | Reflect как triage; атрибуция через текстовые маркеры |
| Reflect как коуч | v0.4.14 | ✅ Отлично | Генерирует полные скрипты/планы из синтеза shared знаний |
| Knowledge Conflicts | v0.4.14 | ✅ С настройкой | skepticism=4-5 для честного разрешения конфликтов |
| PydanticAI | v0.4.15 | 📦 Готов | Пакет в репо, async-native tools + memory_instructions |
| timestamp="unset" | v0.4.15 | ✅ Исправлено | Пропатчено в patched (commit `eb0ee5bd`), event_date nullable |
| Recall perf | v0.4.15 | ✅ Значительно | −50% latency на крупных банках (DB indexes + single query) |
| Reflect overflow | v0.4.15 | ✅ Исправлено | Token budget guard, force-synthesis при превышении |
| Bank-scoped ACL | v0.4.15 | ✅ Готов | validate_bank_read/write + allowed_bank_ids |
| Tag Groups | v0.4.18 | ✅ Работает | Compound boolean predicates (AND/OR/NOT) для сложных фильтров |
| Observation Scopes | v0.4.18 | ✅ Работает | `combined`/`per_tag`/`all_combinations`/`custom` — гранулярная консолидация |
| Reflect structured output | v0.4.18 | ✅ Работает | `response_schema` для JSON-ответа |
| Retain Strategies | v0.4.19 | ✅ Работает | Именованные конфигурации per-bank, `--strategy` в CLI |
| Verbatim/Chunks modes | v0.4.19 | ✅ Работает | Zero-LLM retain (chunks) и minimal-LLM (verbatim) |
| Adaptive consolidation | v0.4.19 | ✅ Работает | Retry с backoff при failed consolidation |
| Reflect wall-clock timeout | v0.4.20 | ✅ Работает | 300s default, prevents runaway agentic loops |
| fact_types / exclude_mental_models | v0.4.20 | ✅ Работает | Гранулярный контроль источников при reflect |
| Delta Retain | v0.4.21 | ✅ Работает | Content-hashing skips LLM для неизменённых чанков при upsert |
| Audit Logging | v0.4.21 | ✅ Готов | Fire-and-forget, настраиваемый по действиям + auto retention sweep |
| LiteLLM provider | v0.4.21 | ✅ Готов | 100+ бэкендов (Bedrock, Azure и др.) через litellm SDK |
| Mental Model Tag Filtering | v0.4.22 | ✅ Работает | trigger.tags_match + tag_groups решают blind spot при refresh |
| Document Metadata API | v0.4.22 | ✅ Работает | metadata field в document list/get endpoints |
| Custom LLM params (EXTRA_BODY) | v0.4.22 | ✅ Работает | Provider-specific params в LLM-запросах |
| Experience classification fix | v0.4.22 | ✅ Исправлено | Текст от первого лица → experience fact_type |
| Bank Templates | v0.5.0 | ✅ Готов | Export/import config + mental models + directives как JSON manifest |
| Retain Append Mode | v0.5.0 | ✅ Готов | `update_mode: "append"` — инкрементальное дополнение документов |
| Mental Model Detail Levels | v0.5.0 | ✅ Готов | `detail=metadata/content/full` — контроль payload |
| llamacpp LLM | v0.5.0 | ✅ Готов | Offline inference (Gemma 4), auto-download, zero API keys |
| OpenRouter LLM/emb/reranker | v0.5.0 | ✅ Готов | 100+ моделей через единый API key |
| Entity Fanout Cap | v0.5.0 | ✅ Авто | 200 per-entity limit, 84% ускорение на больших графах |
| Clear Memories ≠ Delete Bank | v0.5.0 | ✅ Исправлено | DELETE /memories сохраняет bank profile |
| Proof Count Boost | v0.5.0 | ✅ Авто | Факты с 50+ proofs ранжируются выше |
| Sync Retain (MCP) | v0.5.0 | ✅ Готов | Синхронный retain без polling для MCP-клиентов |

---

## CLI Quick Reference

```bash
# Daemon
hindsight-embed daemon start                  # default profile (порт 8888)
hindsight-embed -p sup daemon start           # sup profile (порт 8890)
hindsight-embed daemon status

# Банки
hindsight-embed bank list
hindsight-embed bank disposition <bank>
hindsight-embed bank update <bank> --skepticism 5
hindsight-embed bank stats <bank>
hindsight-embed bank graph <bank>

# Retain (память — запись)
hindsight-embed memory retain <bank> "content"               # базовый
hindsight-embed memory retain <bank> - < file.txt            # stdin
hindsight-embed memory retain <bank> "content" --doc-id uid  # идемпотентный
hindsight-embed memory retain <bank> "content" --async       # неблокирующий
hindsight-embed memory retain <bank> "content" --strategy documents  # именованная стратегия (v0.4.19)

# Recall (память — поиск)
hindsight-embed memory recall <bank> "query"
hindsight-embed memory recall <bank> "query" --tags type:adr           # фильтр по тегам
hindsight-embed memory recall <bank> "query" --tags-match all_strict   # строгий matching
hindsight-embed memory recall <bank> "query" --tag-groups '<json>'     # compound filter (v0.4.18)

# Reflect (аналитическое рассуждение)
hindsight-embed memory reflect <bank> "query"
hindsight-embed memory reflect <bank> "query" --context "текущая ситуация"
hindsight-embed memory reflect <bank> "query" --tags type:security
hindsight-embed memory reflect <bank> "query" --tag-groups '<json>'    # compound filter (v0.4.18)
hindsight-embed memory reflect <bank> "query" -b mid                   # budget: low/mid/high

# Mental Models
hindsight-embed mental-model create <bank> "<name>" "<source_query>"
hindsight-embed mental-model list <bank>
hindsight-embed mental-model get <bank> <id>
hindsight-embed mental-model refresh <bank> <id>

# Sup профиль — добавить -p sup ко всем командам
hindsight-embed -p sup memory retain <bank> "content"
hindsight-embed -p sup memory reflect <bank> "query"

# API-only (нет CLI-эквивалента)
# Entity graph
curl -s "http://localhost:8888/v1/default/banks/{bank_id}/entities"

# Entity labels config
curl -X PATCH "http://localhost:8888/v1/default/banks/{bank_id}/config" \
  -H "Content-Type: application/json" -d '{...}'

# Bank config (read)
curl -s "http://localhost:8888/v1/default/banks/{bank_id}/config"

# Retain strategies config
curl -X PATCH "http://localhost:8888/v1/default/banks/{bank_id}/config" \
  -d '{"updates": {"retain_strategies": {...}}}'

# Reflect с structured output (API-only)
curl -X POST ".../banks/{bank_id}/reflect" \
  -d '{"query": "...", "response_schema": {...}}'

# Bank template export/import (v0.5.0)
curl -s ".../banks/{bank_id}/export" | jq .
curl -X POST ".../banks/{bank_id}/import" -d @template.json

# Mental models с detail level (v0.5.0)
curl -s ".../banks/{bank_id}/mental-models?detail=metadata"

# Retain с append mode (v0.5.0)
curl -X POST ".../banks/{bank_id}/memories/retain" \
  -d '{"items": [{"content": "new text", "document_id": "doc1"}], "update_mode": "append"}'
```

---

## 14. Bank Templates, Append Mode, и другие фичи v0.5.0

### 11.1. Bank Templates — версионирование конфигурации банков

**Что это:** Export/import полной конфигурации банка (config overrides, mental models, directives) как JSON manifest.

**Паттерны:**

| # | Паттерн | Описание |
|---|---------|----------|
| 1 | Template as Code | Экспорт → коммит в git → import на другом instance. Версионирование конфигов |
| 2 | Bank provisioning | Шаблон для нового клиента: disposition + mental models + directives за 1 запрос |
| 3 | Dev/prod parity | Один template → import на dev и prod, гарантия идентичной конфигурации |
| 4 | Backup/restore | Export перед деструктивными операциями, import для восстановления |

**API:**
```bash
# Export
curl -s "http://localhost:8888/v1/default/banks/my-bank/export" > template.json

# Import (создаёт bank если не существует)
curl -X POST "http://localhost:8888/v1/default/banks/new-bank/import" \
  -H "Content-Type: application/json" -d @template.json

# Валидация
curl -s "http://localhost:8888/v1/bank-template-schema" | jq .
```

**Template manifest format:**
```json
{
  "config": { "retain_strategy": "default", "llm_model": "..." },
  "mental_models": [
    { "id": "uuid", "name": "Domain Knowledge", "content": "..." }
  ],
  "directives": [
    { "text": "Always respond in Russian", "priority": 1 }
  ]
}
```

### 11.2. Retain Append Mode — инкрементальное дополнение документов

**Что это:** `update_mode: "append"` конкатенирует новый контент к существующему документу, затем переобрабатывает весь документ. Delta retain автоматически пропускает старые chunks.

**Когда использовать:**
- Накопление сообщений чата в один документ
- Добавление новых заметок к существующей теме
- Логирование последовательных событий

**API:**
```bash
# Первый retain — создаёт документ
curl -X POST ".../banks/{id}/memories/retain" \
  -d '{"items": [{"content": "msg 1", "document_id": "chat-123"}]}'

# Второй retain — дополняет
curl -X POST ".../banks/{id}/memories/retain" \
  -d '{"items": [{"content": "msg 2", "document_id": "chat-123"}], "update_mode": "append"}'
# Результат: документ содержит "msg 1\n\nmsg 2", но LLM обрабатывает только новый chunk
```

**Паттерн для чата:**
- Один `document_id` = одна сессия чата
- Каждое сообщение → `append` retain
- Delta retain хеширует chunks → LLM не переобрабатывает старые сообщения
- Consolidation агрегирует наблюдения из всех сообщений

### 11.3. Mental Model Detail Levels — оптимизация payload

**Что это:** Параметр `detail` на GET/LIST mental models endpoints для контроля объёма возвращаемых данных.

| detail | Что включает | Размер | Когда |
|--------|-------------|--------|-------|
| `metadata` | id, name, created_at, updated_at | ~100 bytes | Boot flow, sidebar list |
| `content` | + text content | ~2-10 KB | Preview, display |
| `full` | + history, trigger config, tags | ~5-50 KB | Admin, edit mode |

**API:**
```bash
# Быстрая загрузка списка (только имена)
curl -s ".../banks/{id}/mental-models?detail=metadata"

# Полные данные конкретной модели
curl -s ".../banks/{id}/mental-models/{mm_id}?detail=full"
```

### 11.4. Новые LLM/Embeddings/Reranker провайдеры

| Провайдер | Тип | Модель по умолчанию | Применение |
|-----------|-----|---------------------|------------|
| `llamacpp` | LLM | Gemma 4 E2B Q4_K_M (~3.5 GB) | Offline dev, zero-cost inference |
| `openrouter` | LLM | qwen/qwen3.5-9b | Доступ к 100+ моделям через один API key |
| `gemini` | Embeddings | gemini-embedding-001 (768d) | Google ecosystem |
| `openrouter` | Embeddings | pplx-embed-v1-0.6b | OpenRouter unified key |
| `google` | Reranker | semantic-ranker-default-004 | Cloud reranking |
| `openrouter` | Reranker | cohere/rerank-v3.5 | OpenRouter unified key |

**llamacpp для dev:** Первый запуск автоматически скачивает модель. Не нужны API ключи. Идеально для тестирования.
```bash
export HINDSIGHT_API_LLM_PROVIDER=llamacpp
# Всё! При первом retain модель скачается и запустится
```

### 11.5. Graph Retrieval Performance — entity fanout cap

**Что изменилось:**
- BFS и MPFP graph retrieval стратегии **удалены** (были deprecated)
- Остаётся только `link_expansion` — единственная graph retrieval стратегия
- **Entity fanout cap:** `LINK_EXPANSION_PER_ENTITY_LIMIT=200` (default)
  - На банках с 25K+ mentions per entity: **84% ускорение recall** (11s → 1.8s)
  - Timeout: `LINK_EXPANSION_TIMEOUT=10.0` сек (предотвращает зависание)

**Когда тюнить:**
- Для высокочастотных entities (системные имена, общие термины) — уменьшить до 50-100
- Для точных domains — увеличить до 500+

### 11.6. Sync Retain (MCP) — синхронный retain без polling

**Что это:** MCP tool `sync_retain` выполняет retain синхронно (без async operation + polling). Полезно для MCP-клиентов (Claude Code, Cursor) где polling неудобен.

### 11.7. Clear Memories — сохранение bank profile

**Что изменилось:** `DELETE /memories` теперь **не удаляет** bank profile (disposition, background, config). Раньше удалял весь банк. Параметр `delete_bank_profile=false` (default).

### 11.8. Proof Count Boost — ранжирование по reliability

**Что это:** Факты с большим количеством supporting evidence (proof_count) ранжируются выше при recall. ~5% boost на факты с 50+ proofs. Автоматически, без конфигурации.
