#!/usr/bin/env python3
"""
hindsight-smart-stop.py
Умный Stop hook — извлекает и сохраняет только значимые факты из сессии.

Что сохраняем:
- Learnings (что узнали нового)
- Решения (architectural decisions)
- Неудачи и их причины
- Процедуры (что сработало)

Что НЕ сохраняем:
- Рутинные операции
- Код (он в git)
- Общие вопросы без выводов
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

# === GRACEFUL DEGRADATION ===
# Никогда не блокируем Claude — любая ошибка = тихий exit
def graceful_exit(signum=None, frame=None):
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

# Конфигурация
BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "hindsight-dev")
MIN_SESSION_TURNS = 4  # Минимум сообщений для анализа
MAX_TRANSCRIPT_CHARS = 15000  # Лимит для анализа

# Путь к локальному hindsight-embed
HINDSIGHT_EMBED = os.environ.get(
    "HINDSIGHT_EMBED_PATH",
    "/Users/andreymiroshkin/hindsight-dev/patched/.venv/bin/hindsight-embed"
)


def check_hindsight_available() -> bool:
    """Проверяем что hindsight-embed доступен."""
    path = Path(HINDSIGHT_EMBED)
    return path.exists() and os.access(path, os.X_OK)

# Паттерны значимого контента (для быстрой проверки без LLM)
SIGNIFICANT_PATTERNS = [
    "решил", "решение", "выбрал", "выбор",
    "проблема", "ошибка", "не работает", "failed", "error",
    "исправил", "fix", "fixed", "починил",
    "узнал", "понял", "learned", "insight",
    "лучше", "хуже", "оптимально", "рекомендую",
    "архитектура", "подход", "strategy",
    "важно", "критично", "запомни", "remember",
    "не делай", "избегай", "avoid", "don't",
    "сработало", "worked", "успешно", "success"
]


def read_input():
    """Читаем JSON из stdin."""
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}


def extract_assistant_messages(transcript_path: str) -> list[str]:
    """Извлекаем сообщения assistant из транскрипта."""
    messages = []
    try:
        with open(transcript_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    turn = json.loads(line)
                    if turn.get("role") == "assistant":
                        content = turn.get("content", "")
                        if isinstance(content, list):
                            # Извлекаем текст из content blocks
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    messages.append(block.get("text", ""))
                        elif isinstance(content, str):
                            messages.append(content)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return messages


def has_significant_content(text: str) -> bool:
    """Быстрая проверка на наличие значимого контента."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in SIGNIFICANT_PATTERNS)


def extract_learnings_with_reflect(content: str, cwd: str) -> str | None:
    """Используем hindsight reflect для извлечения learnings."""
    project_name = Path(cwd).name if cwd else "project"

    query = f"""Проанализируй эту сессию работы над проектом '{project_name}' и извлеки ТОЛЬКО значимые факты для долгосрочной памяти.

Извлеки (если есть):
1. LEARNINGS — что нового узнали, инсайты
2. DECISIONS — принятые решения и их причины
3. FAILURES — что не сработало и почему
4. PROCEDURES — процедуры которые сработали

НЕ включай:
- Рутинные операции (создал файл, запустил тест)
- Код (он в git)
- Общие вопросы без выводов

Если значимых фактов нет — ответь "НЕТ ЗНАЧИМЫХ ФАКТОВ".
Иначе — кратко перечисли факты (по 1 предложению каждый)."""

    try:
        result = subprocess.run(
            [HINDSIGHT_EMBED, "memory", "reflect", BANK_ID, query,
             "-c", content[:MAX_TRANSCRIPT_CHARS],
             "-b", "low", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            answer = data.get("text", "")
            if answer and "НЕТ ЗНАЧИМЫХ ФАКТОВ" not in answer.upper():
                return answer
    except Exception:
        pass
    return None


def retain_learnings(learnings: str, cwd: str):
    """Сохраняем learnings в hindsight (async)."""
    project_name = Path(cwd).name if cwd else "project"

    try:
        subprocess.Popen(
            [HINDSIGHT_EMBED, "memory", "retain", BANK_ID, learnings,
             "--context", f"session-learnings-{project_name}",
             "--async", "-o", "json"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def main():
    try:
        # Проверяем доступность hindsight-embed
        if not check_hindsight_available():
            sys.exit(0)

        input_data = read_input()

        # КРИТИЧНО: предотвращаем infinite loop
        if input_data.get("stop_hook_active"):
            sys.exit(0)

        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", "")

        if not transcript_path or not Path(transcript_path).exists():
            sys.exit(0)

        # Извлекаем сообщения assistant
        messages = extract_assistant_messages(transcript_path)

        # Проверяем минимальное количество
        if len(messages) < MIN_SESSION_TURNS:
            sys.exit(0)

        # Объединяем для анализа
        combined = "\n---\n".join(messages[-10:])  # Последние 10 сообщений

        # Быстрая проверка на значимый контент
        if not has_significant_content(combined):
            sys.exit(0)

        # Извлекаем learnings через reflect
        learnings = extract_learnings_with_reflect(combined, cwd)

        if learnings:
            retain_learnings(learnings, cwd)

    except Exception:
        # Graceful degradation: любая ошибка = тихий exit
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
