# Benchmarks

This directory contains benchmark evaluations for the Entity-Aware Memory System.

## LoComo Benchmark

**Location**: `locomo/`

**Purpose**: Evaluate long-term conversational memory through Question Answering on multi-session conversations.

### Quick Start

1. **Run full benchmark** (10 conversations, ~2000 questions):
   ```bash
   cd locomo
   uv run python run_benchmark.py
   ```

2. **Run quick test** (1 conversation, 10 questions):
   ```bash
   cd locomo
   uv run python run_benchmark.py --max-conversations 1 --max-questions 10
   ```

3. **View results**:
   - Detailed report: `locomo/RESULTS.md`
   - Raw data: `locomo/benchmark_results.json`

### Dataset

- **Source**: [Snap Research LoComo](https://github.com/snap-research/locomo)
- **File**: `locomo10.json` (10 conversations)
- **Size**: Each conversation has ~300 turns over ~35 sessions spanning several months
- **Tasks**: Question Answering with 3 reasoning types (single-hop, temporal, multi-hop)

### Methodology

1. **Ingest** each conversation turn-by-turn with timestamps
2. **Apply** coreference resolution and entity extraction
3. **Create** temporal, semantic, and entity links
4. **Answer** questions using spreading activation search
5. **Evaluate** using LLM-as-judge (GPT-4o-mini)

### Expected Performance

Based on published results:
- **Human**: ~95%
- **Letta (GPT-4o-mini)**: 74.0%
- **Mem0 Graph**: 68.5%
- **Our target**: 65-75% (competitive with state-of-the-art)

### Computational Cost

**Per conversation** (~300 turns):
- ~300 embedding API calls (ingestion)
- ~200 embedding API calls (queries)
- ~200 LLM API calls (answer generation)
- ~200 LLM API calls (judgment)

**Estimated runtime**: 2-5 minutes per conversation (API-dependent)

**Estimated cost**: $0.50-1.00 per conversation (OpenAI pricing)

## LongMemEval Benchmark

**Location**: `longmemeval/`

**Purpose**: Evaluate five core long-term interactive memory abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.

### Quick Start

1. **Download dataset**:
   ```bash
   cd longmemeval
   curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o longmemeval_s_cleaned.json
   ```

2. **Run full benchmark** (500 questions):
   ```bash
   cd longmemeval
   uv run python run_benchmark.py
   ```

3. **Run quick test** (5 instances):
   ```bash
   cd longmemeval
   uv run python run_benchmark.py --max-instances 5
   ```

4. **View results**:
   - Raw data: `longmemeval/benchmark_results.json`

### Dataset

- **Source**: [LongMemEval (ICLR 2025)](https://github.com/xiaowu0162/LongMemEval)
- **File**: `longmemeval_s_cleaned.json` (500 instances)
- **Size**: ~40 sessions per instance (~115k tokens)
- **Tasks**: 5 memory abilities across different question types

### Methodology

1. **Ingest** multi-session conversations with timestamps
2. **Apply** coreference resolution and entity extraction
3. **Create** temporal, semantic, and entity links
4. **Retrieve** relevant memories using spreading activation
5. **Generate** answers using GPT-4o-mini
6. **Evaluate** using GPT-4o as judge

### Expected Performance

Based on published results:
- **Human**: ~95%
- **Zep**: 75.2%
- **Letta (GPT-4o-mini)**: 74.0%
- **Mem0 Graph**: 68.5%
- **Our target**: 65-75% (competitive with state-of-the-art)

### Computational Cost

**Full benchmark** (500 instances):
- Embeddings: Free (local model)
- Answer generation: 500 × GPT-4o-mini calls
- Evaluation: 500 × GPT-4o calls
- **Estimated runtime**: 2-4 hours
- **Estimated cost**: $50-80 (OpenAI API)

## Future Benchmarks

- **MemGPT Tasks**: Long-context question answering
- **Custom Temporal Reasoning**: Time-based memory retrieval
- **Entity-Centric Queries**: Testing entity link effectiveness

## Adding New Benchmarks

1. Create a new directory: `benchmarks/{benchmark_name}/`
2. Add dataset: `benchmarks/{benchmark_name}/data/`
3. Implement adapter: `benchmarks/{benchmark_name}/run_benchmark.py`
4. Document results: `benchmarks/{benchmark_name}/RESULTS.md`

## Results Summary

| Benchmark | Metric | Our System | Best Published | Status |
|-----------|--------|------------|----------------|--------|
| LoComo QA | Accuracy | {TBD}% | 74.0% (Letta) | In Progress |
| LongMemEval | Accuracy | {TBD}% | 75.2% (Zep) | Ready to Run |

*Last updated: 2025-10-30*
