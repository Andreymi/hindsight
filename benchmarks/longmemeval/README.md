# LongMemEval Benchmark

**Purpose**: Evaluate long-term interactive memory capabilities across five core abilities.

## Overview

LongMemEval is a comprehensive benchmark that tests chat assistants on realistic long-term memory scenarios. The benchmark evaluates five core memory abilities:

1. **Information Extraction** - Retrieving specific facts from conversation history
2. **Multi-Session Reasoning** - Connecting information across multiple conversations
3. **Temporal Reasoning** - Understanding time-based relationships and changes
4. **Knowledge Updates** - Handling conflicting or updated information
5. **Abstention** - Recognizing when information is insufficient to answer

## Dataset

- **Source**: [LongMemEval (ICLR 2025)](https://github.com/xiaowu0162/LongMemEval)
- **File**: `longmemeval_s_cleaned.json`
- **Size**: 500 question-answer pairs
- **Context**: ~40 sessions per instance (~115k tokens)
- **Format**: Multi-turn conversations with timestamped sessions

### Dataset Structure

Each instance contains:
- `question_id`: Unique identifier
- `question_type`: Category (single-session, multi-session, temporal, knowledge-update, abstention)
- `question`: Query text
- `answer`: Expected answer
- `question_date`: Query timestamp
- `haystack_sessions`: List of conversation sessions with turns
- `answer_session_ids`: Evidence session identifiers

## Quick Start

### Prerequisites

1. Python 3.11+ with dependencies installed (`uv sync`)
2. PostgreSQL database configured
3. OpenAI API key set in environment

```bash
export OPENAI_API_KEY="your-api-key"
```

### Download Dataset

```bash
cd benchmarks/longmemeval
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o longmemeval_s_cleaned.json
```

### Run Benchmark

**Full evaluation** (500 questions):
```bash
uv run python run_benchmark.py
```

**Quick test** (first 5 instances):
```bash
uv run python run_benchmark.py --max-instances 5
```

**Custom settings**:
```bash
uv run python run_benchmark.py \
  --max-instances 10 \
  --thinking-budget 100 \
  --top-k 20 \
  --output my_results.json
```

### View Results

Results are saved to `benchmark_results.json` and include:
- Per-question scores and predictions
- Performance breakdown by question type
- Retrieved memory units for debugging
- Evaluation explanations

## Methodology

### 1. Ingestion Phase

For each instance:
1. Parse all conversation sessions with timestamps
2. Process each turn (user and assistant messages)
3. Store in memory system with:
   - Coreference resolution (pronouns → entities)
   - Entity extraction and disambiguation
   - Temporal, semantic, and entity link creation

### 2. Retrieval Phase

For each question:
1. Generate query embedding
2. Find entry points (top-3 similar memories)
3. Spread activation through memory graph
4. Apply recency and frequency weighting
5. Return top-k most relevant memory units

### 3. Answer Generation

1. Format retrieved memories as context
2. Generate answer using GPT-4o-mini
3. Enforce answering only from provided memories
4. Handle abstention cases appropriately

### 4. Evaluation

1. Compare predicted answer to gold answer
2. Use GPT-4o as judge for semantic equivalence
3. Binary scoring (1 = correct, 0 = incorrect)
4. Aggregate by question type

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-instances` | 500 | Number of instances to evaluate |
| `--max-questions` | None | Limit questions per instance (for testing) |
| `--thinking-budget` | 100 | Exploration budget for spreading activation |
| `--top-k` | 20 | Number of memory units to retrieve |
| `--output` | `benchmark_results.json` | Output file path |

## Expected Performance

Based on published results:

| System | Accuracy |
|--------|----------|
| Human | ~95% |
| Zep | 75.2% |
| Letta (GPT-4o-mini) | 74.0% |
| Mem0 Graph | 68.5% |
| **Target** | **65-75%** |

## Performance by Question Type

Expected breakdown:

- **Single-session**: 70-80% (easiest - information in one session)
- **Multi-session**: 60-70% (requires connecting across sessions)
- **Temporal reasoning**: 60-70% (requires time-based reasoning)
- **Knowledge updates**: 50-65% (hardest - handling conflicting info)
- **Abstention**: 65-75% (recognizing insufficient information)

## Computational Cost

**Per instance** (~40 sessions, ~200 turns):
- Ingestion: ~200 embedding generations (local model, fast)
- Query: 1 embedding generation + graph search
- Answer: 1 GPT-4o-mini call (~200 tokens)
- Evaluation: 1 GPT-4o call (~150 tokens)

**Full benchmark** (500 instances):
- Runtime: 2-4 hours (depends on API rate limits)
- Cost: ~$50-80 (OpenAI API for answer generation + evaluation)
- Embeddings: Free (local model)

## Example Output

```
LongMemEval Benchmark Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Performance
┏━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric         ┃ Value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total          │ 500   │
│ Correct        │ 345   │
│ Incorrect      │ 155   │
│ Accuracy       │ 69.0% │
└────────────────┴───────┘

Performance by Question Type
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Question Type      ┃ Total ┃ Correct ┃ Accuracy ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ single-session     │ 150   │ 115     │ 76.7%    │
│ multi-session      │ 120   │ 78      │ 65.0%    │
│ temporal-reasoning │ 100   │ 65      │ 65.0%    │
│ knowledge-update   │ 80    │ 48      │ 60.0%    │
│ abstention         │ 50    │ 39      │ 78.0%    │
└────────────────────┴───────┴─────────┴──────────┘
```

## Troubleshooting

### Dataset not found
```bash
cd benchmarks/longmemeval
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o longmemeval_s_cleaned.json
```

### OpenAI API key error
```bash
export OPENAI_API_KEY="your-api-key"
```

### Memory ingestion slow
- This is expected for first-time entity resolution
- Subsequent queries are fast (graph search)
- Consider using `--max-instances` for quick testing

### Low accuracy
- Try increasing `--thinking-budget` (default: 100)
- Try increasing `--top-k` (default: 20)
- Check retrieved memories in results JSON for debugging

## Architecture Integration

This benchmark tests the full memory system architecture:

1. ✅ **Coreference Resolution**: Makes memories self-contained
2. ✅ **Entity Extraction**: Identifies people, organizations, places
3. ✅ **Entity Disambiguation**: Links mentions across sessions
4. ✅ **Temporal Links**: Connects memories by time proximity
5. ✅ **Semantic Links**: Connects memories by meaning
6. ✅ **Entity Links**: Connects memories by shared entities
7. ✅ **Spreading Activation**: Graph-aware retrieval
8. ✅ **Recency/Frequency Weighting**: Importance signals

## Citation

If you use this benchmark, please cite:

```bibtex
@inproceedings{wu2025longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and Wang, Hongwei and Liu, Wenhao and Wang, Jiaheng and Li, Zihan and Huang, Yiqin and Patel, Zelin and Liu, Yiheng and Meng, Bo and Pan, Sinong and others},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Related Benchmarks

- **LoComo**: Multi-session conversational QA
- **MemGPT Tasks**: Long-context question answering
- **Custom Temporal Reasoning**: Time-based memory retrieval
