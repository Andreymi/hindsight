# Exa CLI на Rust — Идеи

## Мотивация

MCP-сервер exa добавляет оверхед:
- `npx mcp-remote` запускает Node.js каждый раз
- SSE соединение держится открытым
- ~2-5 сек на инициализацию

CLI решает это: один бинарник, мгновенный старт.

---

## Актуальное Exa API (январь 2026)

### Endpoints

| Метод | Описание |
|-------|----------|
| `POST /search` | Основной поиск |
| `POST /contents` | Контент по URL |
| `POST /find-similar` | Похожие страницы |
| `POST /answer` | Ответ с цитатами |

### Search Types
- `auto` (default) — умный выбор
- `neural` — embeddings-based
- `fast` — быстрый
- `deep` — глубокий с query expansion

### Categories
- `company`, `research paper`, `news`, `tweet`
- `personal site`, `financial report`, `people`

### Research API (async)
- `research.create({ instructions, model, outputSchema })`
- `research.get(id)` / `research.pollUntilFinished(id)`
- Models: `exa-research`, `exa-research-pro`

---

## CLI Design

### Команды

```bash
# Поиск
exa search "query" [options]
exa s "query"  # alias

# Ответ с цитатами
exa answer "question" [options]
exa a "question"  # alias

# Похожие страницы
exa similar <url> [options]

# Контент из URL
exa contents <url>... [options]

# Исследование (async)
exa research "instructions" [--wait]
exa research status <id>
exa research list

# Код/документация
exa code "query" [--tokens 5000]

# Конфигурация
exa config set api-key <key>
exa config show
exa status
```

### Глобальные флаги

```
-k, --api-key <key>     API ключ (приоритет над env/config)
-o, --output <format>   json | text | markdown
-n, --num <N>           Количество результатов (default: 10)
--raw                   Сырой JSON без форматирования
--debug                 Debug output в stderr
```

### Флаги search

```
-t, --type <type>       auto | neural | fast | deep
-c, --category <cat>    company | news | people | research paper | ...
--domains <list>        Только эти домены
--exclude <list>        Исключить домены
--start-date <date>     После этой даты (ISO 8601)
--end-date <date>       До этой даты
--text                  Включить текст страниц
--highlights            Включить highlights
--summary               Включить summary
```

### Флаги answer

```
--model <model>         exa | exa-pro
--stream                Streaming output
--system <prompt>       System prompt
```

---

## Структура проекта

```
exa-cli/
├── Cargo.toml
├── src/
│   ├── main.rs           # Entry point, clap setup
│   ├── cli.rs            # CLI argument definitions
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── search.rs
│   │   ├── answer.rs
│   │   ├── similar.rs
│   │   ├── contents.rs
│   │   ├── research.rs
│   │   ├── code.rs
│   │   └── config.rs
│   ├── api/
│   │   ├── mod.rs
│   │   ├── client.rs     # reqwest HTTP client
│   │   ├── types.rs      # Request/Response structs
│   │   └── error.rs      # API errors
│   ├── config.rs         # ~/.config/exa/config.json
│   └── output.rs         # Форматирование (colored crate)
```

---

## Dependencies (Cargo.toml)

```toml
[package]
name = "exa-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
colored = "2"
dirs = "5"                    # XDG paths
anyhow = "1"                  # Error handling
thiserror = "1"               # Custom errors
futures = "0.3"               # For streaming
indicatif = "0.17"            # Progress bars

[profile.release]
lto = true
strip = true
codegen-units = 1
```

---

## Config

Путь: `~/.config/exa/config.json`

```json
{
  "api_key": "xxx",
  "default_num_results": 10,
  "default_output": "text"
}
```

Приоритет:
1. `--api-key` флаг
2. `EXA_API_KEY` env var
3. config file

---

## Exit Codes

```rust
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_AUTH: i32 = 2;      // API key missing/invalid
pub const EXIT_TIMEOUT: i32 = 3;
pub const EXIT_NOT_FOUND: i32 = 4;
```

---

## Output Formats

### Text (default)
```
Found 5 results

[1] Title of the page
    https://example.com/page
    Preview of the content...

[2] Another title
    ...
```

### JSON (`--output json`)
Сырой ответ API

### Markdown (`--output md`)
```markdown
## Results for "query"

1. **[Title](https://example.com)**
   > Preview text...

2. **[Another](https://example.com/2)**
   > More text...
```

---

## Streaming (answer --stream)

```rust
// Server-Sent Events
async fn stream_answer(query: &str) -> Result<()> {
    let mut stream = client.stream_answer(query).await?;

    while let Some(chunk) = stream.next().await {
        match chunk? {
            AnswerChunk::Content(text) => {
                print!("{}", text);
                io::stdout().flush()?;
            }
            AnswerChunk::Citations(refs) => {
                println!("\n\nSources:");
                for r in refs {
                    println!("- {}: {}", r.title, r.url);
                }
            }
        }
    }
    Ok(())
}
```

---

## Что взять из tobalsan/exa

✅ Структура команд (search, crawl, research, config, status)
✅ Паттерн `~/.config/exa/config.json`
✅ Exit codes
✅ `--raw` флаг для JSON

❌ Bun/TypeScript (заменить на Rust)
❌ Устаревшие методы API
❌ Отсутствие answer/stream

---

## MVP Scope

Минимум для первой версии:

1. `exa search "query"` — базовый поиск
2. `exa answer "question"` — ответ с цитатами
3. `exa config set api-key` — настройка
4. `exa status` — проверка API

Потом:
- `exa similar`, `exa contents`, `exa code`
- `exa research` с polling
- Streaming для answer
- Shell completions
