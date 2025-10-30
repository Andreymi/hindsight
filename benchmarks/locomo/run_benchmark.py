"""
LoComo Benchmark Runner for Entity-Aware Memory System

Evaluates the memory system on the LoComo (Long-term Conversational Memory) benchmark.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime, timezone, timedelta
from memory import TemporalSemanticMemory
from typing import List, Dict
import openai
from dotenv import load_dotenv
import os
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box

load_dotenv()

console = Console()


def parse_date(date_string: str) -> datetime:
    """Parse LoComo date format to datetime."""
    # Format: "1:56 pm on 8 May, 2023"
    try:
        dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
        return dt.replace(tzinfo=timezone.utc)
    except:
        return datetime.now(timezone.utc)


async def ingest_conversation(memory: TemporalSemanticMemory, conversation_data: Dict, agent_id: str):
    """
    Ingest a LoComo conversation into the memory system (ASYNC version).

    Ingests entire conversation as a single large document for maximum efficiency.

    Args:
        memory: Memory system instance
        conversation_data: Conversation data from LoComo
        agent_id: Agent ID to use
    """
    conv = conversation_data['conversation']
    speaker_a = conv['speaker_a']
    speaker_b = conv['speaker_b']

    # Get all session keys sorted
    session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])

    total_turns = 0

    # Build entire conversation as one large text
    conversation_parts = []

    for session_key in session_keys:
        if session_key not in conv or not isinstance(conv[session_key], list):
            continue

        session_data = conv[session_key]

        # Add all turns from this session
        for turn in session_data:
            speaker = turn['speaker']
            text = turn['text']
            conversation_parts.append(f"{speaker} said: {text}")
            total_turns += 1

    # Ingest entire conversation in ONE put_async call
    # Use the first session date as the event date
    first_session_key = session_keys[0] if session_keys else "session_1"
    date_key = f"{first_session_key}_date_time"
    conversation_date = parse_date(conv.get(date_key, "1:00 pm on 1 January, 2023"))

    full_conversation = " ".join(conversation_parts)

    await memory.put_async(
        agent_id=agent_id,
        content=full_conversation,
        context=f"Full conversation between {speaker_a} and {speaker_b}",
        event_date=conversation_date
    )

    return total_turns


def answer_question(memory: TemporalSemanticMemory, agent_id: str, question: str, thinking_budget: int = 100) -> str:
    """
    Answer a question using the memory system.

    Args:
        memory: Memory system instance
        agent_id: Agent ID
        question: Question to answer
        thinking_budget: How many memory units to explore

    Returns:
        Answer string
    """
    # Search memory
    results = memory.search(
        agent_id=agent_id,
        query=question,
        thinking_budget=thinking_budget,
        top_k=20  # Get more results for better context
    )
    print("question:", question)
    print("Got results:", results)

    if not results:
        return "I don't have enough information to answer that question."

    # Build context from top results
    context_parts = []
    for i, result in enumerate(results[:10], 1):
        context_parts.append(f"{i}. {result['text']}")

    context = "\n".join(context_parts)

    # Use OpenAI to generate answer from context
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't contain the answer, say 'I don't know'."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def evaluate_qa_task(
    memory: TemporalSemanticMemory,
    agent_id: str,
    qa_pairs: List[Dict],
    sample_id: str,
    max_questions: int = None
) -> Dict:
    """
    Evaluate the QA task.

    Returns:
        Dict with evaluation metrics
    """
    results = []

    questions_to_eval = qa_pairs[:max_questions] if max_questions else qa_pairs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating QA for sample {sample_id}...", total=len(questions_to_eval))

        for qa in questions_to_eval:
            question = qa['question']
            correct_answer = qa['answer']
            category = qa.get('category', 0)

            # Get predicted answer
            predicted_answer = answer_question(memory, agent_id, question)

            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'category': category
            })

            progress.update(task, advance=1)

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics.

    Uses LLM-as-judge to evaluate answer quality.
    """
    correct = 0
    total = len(results)

    category_stats = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Judging answers with LLM...", total=total)

        for result in results:
            # Use LLM as judge
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an objective judge. Determine if the predicted answer is semantically equivalent to the correct answer. Answer with ONLY 'yes' or 'no'."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {result['question']}\nCorrect answer: {result['correct_answer']}\nPredicted answer: {result['predicted_answer']}\n\nAre they equivalent?"
                        }
                    ],
                    temperature=0,
                    max_tokens=5
                )

                judgment = response.choices[0].message.content.strip().lower()
                is_correct = 'yes' in judgment

                if is_correct:
                    correct += 1

                result['is_correct'] = is_correct

                # Track by category
                category = result['category']
                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                category_stats[category]['total'] += 1
                if is_correct:
                    category_stats[category]['correct'] += 1

            except Exception as e:
                console.print(f"[red]Error judging answer: {e}[/red]")
                result['is_correct'] = False

            progress.update(task, advance=1)

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'category_stats': category_stats,
        'detailed_results': results
    }


def run_benchmark(max_conversations: int = None, max_questions_per_conv: int = None):
    """
    Run the LoComo benchmark.

    Args:
        max_conversations: Maximum number of conversations to evaluate (None for all)
        max_questions_per_conv: Maximum questions per conversation (None for all)
    """
    console.print("\n[bold cyan]LoComo Benchmark - Entity-Aware Memory System[/bold cyan]")
    console.print("=" * 80)

    # Load dataset
    console.print("\n[1] Loading LoComo dataset...")
    with open('locomo10.json', 'r') as f:
        dataset = json.load(f)

    conversations_to_eval = dataset[:max_conversations] if max_conversations else dataset
    console.print(f"    [green]✓[/green] Loaded {len(conversations_to_eval)} conversations")

    # Initialize memory system
    console.print("\n[2] Initializing memory system...")
    memory = TemporalSemanticMemory()
    console.print("    [green]✓[/green] Memory system initialized")

    # Run evaluation for each conversation
    all_results = []

    for i, conv_data in enumerate(conversations_to_eval, 1):
        sample_id = conv_data['sample_id']
        agent_id = f"locomo_{sample_id}"

        console.print(f"\n[bold blue]Conversation {i}/{len(conversations_to_eval)}[/bold blue] (Sample ID: {sample_id})")

        # Clear previous data
        import psycopg2
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory_units WHERE agent_id = %s", (agent_id,))
        cursor.execute("DELETE FROM memory_links WHERE agent_id = %s", (agent_id,))
        cursor.execute("DELETE FROM entity_cooccurrences WHERE agent_id = %s", (agent_id,))
        cursor.execute("DELETE FROM unit_entities WHERE agent_id = %s", (agent_id,))
        cursor.execute("DELETE FROM entities WHERE agent_id = %s", (agent_id,))
        conn.commit()
        cursor.close()
        conn.close()

        # Ingest conversation (using async for parallel embedding generation)
        console.print("  [3] Ingesting conversation (async with parallel embeddings)...")
        total_turns = asyncio.run(ingest_conversation(memory, conv_data, agent_id))
        console.print(f"      [green]✓[/green] Ingested {total_turns} conversation turns")

        # Evaluate QA
        console.print(f"  [4] Evaluating {len(conv_data['qa'])} QA pairs...")
        qa_results = evaluate_qa_task(
            memory,
            agent_id,
            conv_data['qa'],
            sample_id,
            max_questions=max_questions_per_conv
        )

        # Calculate metrics
        console.print("  [5] Calculating metrics...")
        metrics = calculate_metrics(qa_results)

        console.print(f"      [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")

        all_results.append({
            'sample_id': sample_id,
            'metrics': metrics,
            'total_turns': total_turns
        })

    # Overall results
    console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")

    # Calculate overall metrics
    total_correct = sum(r['metrics']['correct'] for r in all_results)
    total_questions = sum(r['metrics']['total'] for r in all_results)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    # Display results table
    table = Table(title="LoComo Benchmark Results", box=box.ROUNDED)
    table.add_column("Sample ID", style="cyan")
    table.add_column("Turns", justify="right", style="yellow")
    table.add_column("Questions", justify="right", style="blue")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="magenta")

    for result in all_results:
        metrics = result['metrics']
        table.add_row(
            result['sample_id'],
            str(result['total_turns']),
            str(metrics['total']),
            str(metrics['correct']),
            f"{metrics['accuracy']:.1f}%"
        )

    table.add_row(
        "[bold]OVERALL[/bold]",
        "-",
        f"[bold]{total_questions}[/bold]",
        f"[bold]{total_correct}[/bold]",
        f"[bold]{overall_accuracy:.1f}%[/bold]"
    )

    console.print(table)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'conversation_results': all_results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run LoComo benchmark')
    parser.add_argument('--max-conversations', type=int, default=None, help='Maximum conversations to evaluate')
    parser.add_argument('--max-questions', type=int, default=None, help='Maximum questions per conversation')

    args = parser.parse_args()

    results = run_benchmark(
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions
    )

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to benchmark_results.json")
