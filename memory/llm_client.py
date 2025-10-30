"""
LLM client for fact extraction and other AI-powered operations.

Uses OpenAI-compatible API (works with Groq, OpenAI, etc.)
"""
import os
import json
import re
from typing import List, Dict, Optional, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """A single extracted fact from text."""
    fact: str = Field(
        description="Self-contained factual statement with subject + action + context"
    )
    speaker: str = Field(
        default="narrator",
        description="Who said this (name or 'narrator' if not a conversation)"
    )
    type: Literal["biographical", "event", "opinion", "recommendation", "description", "relationship"] = Field(
        description="Category of the fact"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level of the extraction"
    )


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""
    facts: List[ExtractedFact] = Field(
        description="List of extracted factual statements"
    )


def split_into_sentences(text: str) -> List[str]:
    """
    Fast sentence splitter using regex.
    Splits on periods, exclamation marks, and question marks followed by whitespace or end of string.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    # Split on sentence boundaries: .!? followed by space/newline/end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_chars: int = 120000) -> List[str]:
    """
    Split text into chunks at sentence boundaries.

    Keeps chunks under max_chars (~30k tokens assuming 1 token ≈ 4 chars).
    This prevents hitting output token limits on large documents.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)

    Returns:
        List of text chunks, each under max_chars
    """
    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If single sentence exceeds max_chars, split it forcefully
        if sentence_length > max_chars:
            # Save current chunk if any
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split long sentence into smaller pieces
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue

        # If adding this sentence would exceed limit, start new chunk
        if current_length + sentence_length + 1 > max_chars:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def get_llm_client() -> AsyncOpenAI:
    """
    Get configured async LLM client.

    Supports:
    - Groq (default): Set GROQ_API_KEY and optionally GROQ_BASE_URL
    - OpenAI: Set OPENAI_API_KEY

    Returns:
        Configured AsyncOpenAI client
    """
    # Check for Groq configuration first
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        return AsyncOpenAI(
            api_key=groq_api_key,
            base_url=base_url
        )

    # Fall back to OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        return AsyncOpenAI(api_key=openai_api_key)

    raise ValueError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY environment variable."
    )


async def extract_facts_from_text(
    text: str,
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.1,
    max_tokens: int = 65000,
    chunk_size: int = 60000
) -> List[Dict[str, str]]:
    client = get_llm_client()

    # Chunk text if necessary
    chunks = chunk_text(text, max_chars=chunk_size)

    all_facts = []

    for i, chunk in enumerate(chunks):
        prompt = f"""You are extracting facts from text for an AI memory system. Each fact will be stored and retrieved later.

## CRITICAL: Facts must be DETAILED and COMPREHENSIVE

Each fact should:
1. Be SELF-CONTAINED - readable without the original context
2. Include ALL relevant details: WHO, WHAT, WHERE, WHEN, WHY, HOW
3. Preserve specific names, dates, numbers, locations, relationships
4. Resolve pronouns to actual names/entities
5. Include surrounding context that makes the fact meaningful
6. Capture nuances, reasons, causes, and implications

## What to EXTRACT:
- Biographical information (jobs, roles, backgrounds, experiences)
- Events (what happened, when, where, who was involved, why)
- Opinions and beliefs (who believes what and why)
- Recommendations and advice (specific suggestions with reasoning)
- Descriptions (detailed explanations of how things work)
- Relationships (connections between people, organizations, concepts)

## What to SKIP:
- Greetings, thank yous, acknowledgments
- Filler words ("um", "uh", "like")
- Pure reactions without content ("wow", "cool")
- Incomplete thoughts

## EXAMPLES of GOOD facts (detailed, comprehensive):

Input: "Alice mentioned she works at Google in Mountain View. She joined the AI team last year and loves working on large language models."
GOOD: "Alice works at Google in Mountain View on the AI team, which she joined last year, and she loves working on large language models"
BAD: "Alice works at Google" (too short, missing context)

Input: "Bob said he's been hiking every weekend in Yosemite because it helps him clear his mind after stressful work weeks."
GOOD: "Bob has been hiking every weekend in Yosemite because it helps him clear his mind after stressful work weeks"
BAD: "Bob hikes in Yosemite" (missing frequency, reason, and context)

Input: "The new algorithm reduced latency by 40% compared to the baseline by using a novel caching strategy."
GOOD: "The new algorithm reduced latency by 40% compared to the baseline by using a novel caching strategy"
BAD: "The algorithm is faster" (missing numbers, comparison, and method)

## TEXT TO EXTRACT FROM:
{chunk}

Remember: Include ALL details, names, numbers, reasons, and context. Facts should be rich and informative, not summaries."""

        # Use parse() for structured outputs with Pydantic models
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract detailed, comprehensive facts from text. Preserve all context, details, and nuances. Never summarize or shorten - include everything relevant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=FactExtractionResponse
        )

        # Extract the parsed response
        extraction_response = response.choices[0].message.parsed

        # Convert to dict format and add to aggregate
        chunk_facts = [fact.model_dump() for fact in extraction_response.facts]
        all_facts.extend(chunk_facts)

        # Log progress for large documents
        if len(chunks) > 1:
            print(f"Processed chunk {i + 1}/{len(chunks)}: extracted {len(chunk_facts)} facts")

    return all_facts
