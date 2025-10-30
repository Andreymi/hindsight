"""
Test that the improved prompt extracts detailed, comprehensive facts.
"""
import pytest
from memory.llm_client import extract_facts_from_text


@pytest.mark.asyncio
async def test_detailed_extraction_preserves_context():
    """Test that facts preserve all context and details."""

    text = """
    Alice mentioned she works at Google in Mountain View on the AI research team.
    She joined last year after finishing her PhD at Stanford, and she's currently
    focused on improving large language model safety through red teaming and
    adversarial testing. She said the work is challenging but very rewarding because
    it directly impacts millions of users.
    """

    facts = await extract_facts_from_text(text)

    print(f"\nExtracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"{i}. {fact['fact']}")
        print(f"   Type: {fact['type']}, Speaker: {fact['speaker']}, Confidence: {fact['confidence']}\n")

    # Verify we got facts
    assert len(facts) > 0, "Should extract at least one fact"

    # Check that facts contain detailed information
    fact_texts = [f['fact'].lower() for f in facts]
    combined_facts = ' '.join(fact_texts)

    # Should preserve location details
    assert 'mountain view' in combined_facts, "Should preserve specific location 'Mountain View'"

    # Should preserve team/department
    assert 'ai' in combined_facts or 'research' in combined_facts, "Should preserve team information"

    # Should preserve educational background
    assert 'stanford' in combined_facts or 'phd' in combined_facts, "Should preserve educational background"

    # Should preserve work details
    assert 'safety' in combined_facts or 'red teaming' in combined_facts or 'adversarial' in combined_facts, \
        "Should preserve specific work focus details"

    # Check that at least one fact is reasonably detailed (not just "Alice works at Google")
    detailed_fact_found = any(len(f['fact'].split()) >= 10 for f in facts)
    assert detailed_fact_found, "At least one fact should be detailed (10+ words)"


@pytest.mark.asyncio
async def test_numbers_and_metrics_preserved():
    """Test that numbers, percentages, and metrics are preserved."""

    text = """
    Bob explained that the new caching algorithm reduced API latency by 40%
    compared to the baseline, processing 10,000 requests per second instead
    of the previous 7,000. This improvement was achieved by implementing a
    two-tier LRU cache with 1GB memory allocation.
    """

    facts = await extract_facts_from_text(text)

    print(f"\nExtracted {len(facts)} facts:")
    for fact in facts:
        print(f"- {fact['fact']}")

    combined = ' '.join([f['fact'] for f in facts])

    # Should preserve specific numbers
    assert '40' in combined or 'forty' in combined.lower(), "Should preserve percentage"
    assert '10,000' in combined or '10000' in combined or 'ten thousand' in combined.lower(), \
        "Should preserve request rate"
    assert 'cache' in combined.lower(), "Should preserve technical details"


@pytest.mark.asyncio
async def test_reasons_and_causality_preserved():
    """Test that reasons, causes, and explanations are preserved."""

    text = """
    Sarah has been meditating every morning for the past 6 months because
    she found it significantly reduced her anxiety levels and improved her
    focus during work hours. She started this practice after reading a research
    paper on mindfulness benefits.
    """

    facts = await extract_facts_from_text(text)

    print(f"\nExtracted {len(facts)} facts:")
    for fact in facts:
        print(f"- {fact['fact']}")

    combined = ' '.join([f['fact'] for f in facts])

    # Should preserve the causal relationship (because/reason)
    assert any(keyword in combined.lower() for keyword in ['because', 'reduced', 'anxiety', 'improved']), \
        "Should preserve the reason/causality"

    # Should preserve frequency
    assert 'morning' in combined.lower() or 'every' in combined.lower(), \
        "Should preserve frequency information"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
