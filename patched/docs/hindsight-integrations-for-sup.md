# Hindsight: Интеграции и примеры — обзор для Sup

## 1. Готовые интеграции (hindsight-integrations/)

### hindsight-litellm (Python)
- **Что:** Callback для LiteLLM — автоматический recall/retain при каждом LLM-вызове
- **Как подключается:** split API: `configure()` (static settings) → `set_defaults()` (per-call defaults) → `completion(hindsight_query=...)` (вызов с обязательным query)
- **Фишки:** обёртки `wrap_openai()` / `wrap_anthropic()` для нативных SDK, async версии, `get_pending_retain_errors()` для async error tracking, `set_bank_mission()` для контекста извлечения фактов
- **Паттерн:** implicit memory — перехватывает вызов, делает recall, инжектирует в system message, после ответа retain
- **Для sup:** не оптимально (Python), но полезен если нужен отдельный AI-сервис

### hindsight-ai-sdk (TypeScript)
- **Что:** 5 tools для Vercel AI SDK — retain, recall, reflect, getMentalModel, getDocument
- **Как подключается:** `createHindsightTools({ client, bankId })` → передаётся в `generateText()/streamText()`
- **Паттерн:** agent tools — LLM сам решает когда recall/retain/reflect
- **Для sup:** ✅ лучший выбор — TypeScript, совместим с SvelteKit через `@ai-sdk/svelte`

### hindsight-chat (TypeScript)
- **Что:** HOF-обёртка для Vercel Chat SDK (Slack, Discord, Teams, Google Chat, GitHub, Linear)
- **Как подключается:** `withHindsightChat(options, handler)` оборачивает Chat SDK handler
- **Фишки:** авто-resolve bank_id из `msg.author.userId`, `ctx.memoriesAsSystemPrompt()`
- **Для sup:** полезно если sup интегрируется с мессенджерами

### hindsight-crewai (Python)
- **Что:** Storage backend для CrewAI — автоматическая память для multi-agent crews
- **Как подключается:** `Crew(external_memory=ExternalMemory(storage=HindsightStorage(bank_id="...")))`
- **Фишки:** per-agent banks (`per_agent_banks=True`), ReflectTool для явного reflect
- **Для sup:** если будет Python AI-бэкенд с multi-agent системой

### hindsight-openclaw (TypeScript)
- **Что:** Plugin для OpenClaw/MoltBot — автоматическая память для AI-агентов
- **Фишки:** защита от feedback loop, дедупликация recalls, per-user banks по `{platform}-{senderId}`, `bankMission` для контекста, `dynamicBankGranularity` (agent/channel/user/provider), external API mode
- **Для sup:** нерелевантно (специфичная платформа)

### hindsight-pydantic-ai (Python)
- **Что:** Async-native tools для PydanticAI агентов — retain, recall, reflect + auto-inject воспоминаний
- **Как подключается:** `create_hindsight_tools(client, bank_id)` → передаётся в `Agent(tools=[...])`
- **Фишки:** `memory_instructions()` для injection в system prompt, `include_retain/recall/reflect` flags, `tags`/`recall_tags` фильтрация
- **Для sup:** релевантно если бэкенд на Python + PydanticAI

### hindsight-agno (Python)
- **Что:** Memory toolkit для Agno-агентов (native Toolkit pattern)
- **Как подключается:** `AgnoHindsightToolkit(client, bank_id)` → передаётся в Agno Agent
- **Фишки:** динамический bank resolver (`bank_resolver` callable), автоматический per-user bank через `RunContext.user_id`
- **Для sup:** релевантно если бэкенд использует Agno framework

### hindsight-hermes (Python)
- **Что:** Plugin для Hermes Agent через entry point registration
- **Фишки:** три tools (retain, recall, reflect), env var конфигурация, MCP server как альтернатива
- **Для sup:** нишевый use case (Hermes Agent framework)

### Skills Integration (Claude Code / OpenCode / Codex CLI)
- **Что:** Интеграция Hindsight как набора skills для AI-ассистентов разработчика
- **Два режима:** Local (embedded pg0) и Cloud (Hindsight Cloud API)
- **Для sup:** полезно для developer experience, не для production

### MCP Server (встроенный)
- **Что:** Model Context Protocol — AI-ассистенты (Claude Code, Claude Desktop) работают с памятью напрямую
- **28-30 инструментов:** retain, recall, reflect, mental models, directives, banks, documents, operations
  - single-bank mode (`/mcp/{bank_id}/`): **28 tools** (без bank management)
  - multi-bank mode (`/mcp/`): **30 tools** (+ list_banks, create_bank)
- **Новое (v0.4.21):** `filter_mcp_tools` hook (per-user visibility), stateless HTTP mode, `strategy` param на retain
- **Для sup:** полезно для developer experience, не для production чатбота

### hindsight-langgraph (Python) — v0.4.20
- **Что:** Persistent memory для LangGraph и LangChain агентов через 3 паттерна: memory tools, pre-built graph nodes, BaseStore adapter
- **Как подключается:** `create_hindsight_tools(client)` → tools в LangChain agent; или `HindsightRecallNode`/`HindsightRetainNode` в LangGraph
- **Фишки:** async-native (`aretain`, `arecall`, `areflect`), dynamic bank_id per-request через RunnableConfig
- **Для sup:** ❌ Python, но полезен как reference для server-side memory patterns

### hindsight-claude-code (Skills/Hooks) — v0.4.20
- **Что:** Plugin для Claude Code — hook-based auto-capture sessions + recall context
- **Как подключается:** hooks (SessionStart, UserPromptSubmit, Stop, SessionEnd), конфигурация через `settings.json`
- **Фишки:** fire-and-forget async retain, channel-agnostic (Telegram, Discord, Slack), local daemon или external API
- **Для sup:** полезно для dev team tooling, не для production

### hindsight-nemoclaw (TypeScript) — v0.4.20
- **Что:** One-command setup для NemoClaw/OpenClaw sandboxes: `npx @vectorize-io/hindsight-nemoclaw setup`
- **Фишки:** zero code changes, автоматизирует plugin install + config + network policy + gateway restart
- **Для sup:** ❌ специфичная платформа

### hindsight-strands (Python) — v0.4.21
- **Что:** Native @tool functions для AWS Strands Agents SDK — retain, recall, reflect
- **Как подключается:** `create_hindsight_tools(client, bank_id)` → `Agent(tools=[...])`
- **Фишки:** selective tool inclusion, global `configure()` + per-call overrides, `recall_tags_match` modes
- **Для sup:** ❌ Python + AWS-специфичный

### hindsight-ag2 (Python) — v0.4.21
- **Что:** Memory tools для AG2 (AutoGen v2) multi-agent workflows
- **Как подключается:** `register_hindsight_tools(agent, client)` — одна строка
- **Фишки:** GroupChat shared memory, Annotated type hints, metadata + document_id, reflect с response_schema
- **Для sup:** ❌ Python multi-agent framework

### hindsight-llamaindex (Python) — v0.4.21
- **Что:** Два паттерна: HindsightToolSpec (agent-driven tools) + HindsightMemory (BaseMemory auto-memory)
- **Как подключается:** `HindsightToolSpec(client, bank_id)` или `HindsightMemory(client, bank_id)` → agent
- **Фишки:** async-native, automatic background memory (messages stored on every turn)
- **Для sup:** ❌ Python, но BaseMemory pattern полезен как reference для automatic memory injection

### hindsight-codex (Python hooks) — v0.4.21
- **Что:** 3 Python hook скрипта для OpenAI Codex CLI — auto-recall перед prompt + auto-retain после response
- **Фишки:** chunked vs full-session retention, dynamic bank_id per project, structured tool call retention
- **Для sup:** ❌ специфичный CLI

> **Примечание:** С v0.4.20 интеграции имеют **независимое версионирование** — собственные релизные циклы отдельно от core Hindsight API.

---

## 2. Примеры из cookbook

### Chat Memory App (Next.js + Groq)
- Per-user чат с памятью, retain/recall при каждом сообщении
- **Паттерн:** bank-per-browser-session, upsert по document_id

### Taste AI (Next.js + Vercel AI SDK v6)
- Персональный пищевой ассистент
- **Паттерны:** recall + reflect для персонализации, mental models с auto-refresh, directives для языка
- **Изоляция:** single bank + теги `user:${username}`

### Stance Tracker (Next.js + Tavily)
- Отслеживание политических позиций с веб-скрапингом
- **Паттерн:** temporal queries — `recall(bankId, query, { queryTimestamp: '...' })`

### Deliveryman Demo (React + Phaser + FastAPI)
- Обучение агента-курьера через mental models
- **Паттерн:** retain наблюдений → mental model → оптимизация поведения

### Tool Learning Demo (Streamlit)
- LLM учится routing через feedback, хранимый в Hindsight
- **Паттерн:** feedback loop — ошибка → retain → recall при следующем решении

### OpenAI Fitness Coach (Assistants API)
- Фитнес-тренер с bidirectional memory
- **Паттерн:** хранятся и данные пользователя, и мнения/наблюдения коуча

### Support Agent (Jupyter notebook recipe)
- **Паттерн multi-bank:** per-user bank + shared docs bank + learnings bank
- Наиболее релевантен для sup

### CableConnect — AI Customer Service Copilot (v0.4.20+)
- Node.js + web UI, copilot предлагает ответы, CSR подтверждает/отклоняет
- **Паттерн:** feedback loop — interact → retain → improve. Copilot учится на коррекциях
- **Для sup:** ✅ ближайший к sup use case (admin + AI assistant + learning)

### ClaimsIQ — Insurance Claims Triage (v0.4.20+)
- Agent progression от "confused rookie" до "seasoned expert" через memory
- **Паттерн:** dashboard визуализация прогресса обучения

### Chat SDK Multi-Platform Bot (v0.4.21+)
- Slack + Discord бот с единым Hindsight bank через Vercel Chat SDK + AI SDK
- **Для sup:** ✅ показывает cross-channel memory consistency

### Go Memory-Augmented API (v0.4.21+)
- Microservice с per-user memory isolation, fire-and-forget через goroutines
- **Паттерн:** personalized assistance pattern

### Pydantic AI + Memory (v0.4.20+)
- `create_hindsight_tools()` + `memory_instructions()` auto-injection
- **Паттерн:** persistent memory across invocations

### CrewAI + Memory (v0.4.20+)
- Drop-in memory backend, agent learning from prior research
- **Паттерн:** multi-agent shared memory

### Hermes Agent + Memory (v0.4.20+)
- Native plugin через entry point registration, graceful degradation

---

## 3. Сценарии для Sup

### Сценарий A: Чатбот для сотрудников в админке
- **Банки:** `org-{orgId}-knowledge` (shared) + `employee-{orgId}-{userId}` (personal)
- **Интеграция:** hindsight-ai-sdk + Vercel AI SDK + @ai-sdk/svelte
- **MVP:** implicit memory (recall → inject → LLM → retain), потом → agent tools
- **Плюсы:** аудитория уже есть, низкий риск, быстрый time-to-value

### Сценарий B: Публичный чатбот для посетителей
- **Банки:** `visitor-{sessionId}` или `visitor-{email}`
- **Dispositions:** empathy: 5, skepticism: 1
- **Риски:** публичный, нужна модерация, cold start

### Сценарий C: Activity-aware ассистент
- Activities из sup → retain → recall по активностям
- **Статус:** activity passthrough реализован и работает (fire-and-forget)
- **Паттерн:** temporal queries для ретроспективы
- **Новое (v0.4.21):** delta retain оптимизирует re-import логов — только изменённые чанки проходят LLM

### Сценарий D: Sales-бот (уже экспериментировали)
- Shared bank `sales-playbook` + per-seller banks
- Mental models для трекинга клиентов

---

## 4. Сравнение интеграций для Sup

| Критерий | LiteLLM (Python) | Vercel AI SDK (TypeScript) | PydanticAI (Python) |
|----------|-------------------|---------------------------|---------------------|
| Язык | Python | TypeScript | Python |
| Стек Sup | ❌ отдельный сервис | ✅ нативно в SvelteKit | ❌ отдельный сервис |
| Провайдеры | 100+ из коробки | через adapters | многие (OpenAI-compatible) |
| Memory интеграция | Автоматическая (callback) | Явная (tools) или ручная | Явная (tools) + instructions injection |
| Tool calling | ✅ | ✅ лучше типизирован | ✅ Pydantic-validated |
| Streaming | ✅ | ✅ | ✅ |
| Async-native | ✅ | ✅ | ✅ |

**Рекомендация:** Vercel AI SDK + hindsight-ai-sdk — нативно для SvelteKit, не нужен Python-сервис.

---

## 5. Архитектура прототипа (рекомендуемая)

```
┌─────────────────────────────────────┐
│  Sup Admin (SvelteKit)              │
│  ┌───────────────────────┐          │
│  │  Chat Widget (Svelte) │          │
│  │  @ai-sdk/svelte       │          │
│  └──────────┬────────────┘          │
│             │ fetch                  │
│  ┌──────────▼────────────┐          │
│  │  /api/chat endpoint   │          │
│  │  Vercel AI SDK        │          │
│  │  + hindsight-ai-sdk   │          │
│  └──────┬─────────┬──────┘          │
└─────────┼─────────┼─────────────────┘
          │         │
    ┌─────▼───┐ ┌───▼──────────┐
    │ LLM API │ │ Hindsight API│
    └─────────┘ │ (sup profile)│
                └──────────────┘
```

### Два подхода к памяти

**Подход A — Implicit (MVP):**
```
User msg → recall(personal) + recall(shared) → inject в system prompt → streamText() → retain(conversation)
```

**Подход B — Agent tools (production):**
```
User msg → streamText({ tools: hindsightTools }) → агент сам recall/retain/reflect
```

### Новые API-возможности (v0.4.18-v0.4.19)

| Фича | Описание | Релевантность для sup |
|------|----------|----------------------|
| `tag_groups` | Compound boolean filters (AND/OR/NOT) для тегов | ✅ Multi-user фильтрация |
| `observation_scopes` | Per-tag/per-combination консолидация | ✅ Per-user аналитика |
| `retain_strategies` | Именованные конфигурации extract-режимов | ✅ Разный контент (docs vs chats) |
| `verbatim`/`chunks` modes | Zero-LLM retain | ✅ Быстрый import документов |
| `response_schema` в reflect | Structured JSON output | ✅ Интеграция в UI |
| `--context` в reflect | Инъекция текущей ситуации | ✅ Контекст текущей задачи |

### Новые API-возможности (v0.4.20-v0.4.22)

| Фича | Описание | Релевантность для sup |
|------|----------|----------------------|
| **Delta Retain** | Content-hashing, skip LLM для неизменённых чанков при upsert | ✅ Экономия при re-import activity/notes |
| **Reflect wall-clock timeout** | 300s default, предотвращает runaway loops | ✅ Production stability |
| **`fact_types` filter** | Ограничить reflect типами фактов (world/experience) | Полезно для аналитики |
| **`exclude_mental_models`** | Reflect без кэшированных моделей (fresh-only) | Полезно для свежего анализа |
| **Audit Logging** | Fire-and-forget логирование операций | ✅ Usage tracking per tenant |
| **LiteLLM provider** | 100+ LLM-бэкендов (Bedrock, Azure, Together...) | Полезно при смене LLM |
| **Mental Model Tag Filtering** | `trigger.tags_match`/`tag_groups` при refresh | ✅ Per-user mental models |
| **Document Metadata API** | metadata в document list/get endpoints | ✅ Source tracking |
| **Custom LLM params** | `EXTRA_BODY` для provider-specific settings | Полезно при self-hosted |
| **MCP stateless HTTP** | POST-only mode для Claude Code | Dev tooling |

### Вопросы для решения перед стартом
1. LLM провайдер (OpenAI / Anthropic / другой)
2. Где жить чату (страница / floating widget / sidebar)
3. Scope MVP (просто чат / с knowledge base)
4. Hindsight API (sup profile / default pg0)
