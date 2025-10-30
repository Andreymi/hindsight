"""
Test LLM-based fact extraction.
"""
import pytest
from memory.llm_client import extract_facts_from_text
from memory.utils import extract_facts


async def test_fact_extraction_filters_pleasantries():
    """Test that fact extraction filters out social pleasantries."""

    conversation = """
    Host: Welcome to the show, Marta! Thanks for joining us.
    Marta: Oh, thank you so much for having me!
    Host: So tell us, what do you do?
    Marta: I work at Google as a software engineer. I've been there for 3 years now.
    Host: That's amazing!
    Marta: Yeah, I really enjoy it. I mostly work on AI infrastructure.
    Host: Uh-huh, interesting.
    Marta: And I'm also passionate about hiking. I go to Yosemite almost every weekend.
    Host: Wow, that sounds great!
    """

    facts = await extract_facts_from_text(conversation)

    # Extract just the fact texts
    fact_texts = [f['fact'].lower() for f in facts]

    print("\nExtracted facts:")
    for fact in facts:
        print(f"  - {fact['fact']} (speaker: {fact['speaker']}, type: {fact['type']})")

    # Should extract meaningful facts
    assert any('google' in fact and 'software engineer' in fact for fact in fact_texts), \
        "Should extract Marta's job at Google"
    assert any('yosemite' in fact and 'hiking' in fact for fact in fact_texts), \
        "Should extract Marta's hiking hobby"

    # Should NOT extract pleasantries
    assert not any('thank you' in fact for fact in fact_texts), \
        "Should not extract 'thank you'"
    assert not any('amazing' in fact and len(fact.split()) < 5 for fact in fact_texts), \
        "Should not extract simple reactions like 'that's amazing'"
    assert not any('uh-huh' in fact for fact in fact_texts), \
        "Should not extract acknowledgments"


async def test_fact_extraction_makes_self_contained():
    """Test that facts are self-contained (pronouns resolved)."""

    conversation = """
    Alice told me she works at Microsoft.
    She mentioned that she's been there for 5 years.
    She really enjoys her team.
    """

    facts = await extract_facts_from_text(conversation)

    print("\nExtracted facts:")
    for fact in facts:
        print(f"  - {fact['fact']}")

    # All facts should mention "Alice" explicitly, not "she"
    for fact in facts:
        fact_text = fact['fact'].lower()
        # If it's about Alice, it should say "alice" not "she"
        if 'microsoft' in fact_text or 'team' in fact_text:
            assert 'alice' in fact_text, \
                f"Fact should be self-contained with 'Alice', not pronouns: {fact['fact']}"


async def test_extract_facts_util_function():
    """Test the utils.extract_facts() wrapper function."""

    text = """
    Bob is a chef in New York. He owns a restaurant called "The Kitchen".
    Thank you! Yeah, uh-huh.
    """

    facts = await extract_facts(text)

    print("\nExtracted facts:")
    for fact in facts:
        print(f"  - {fact}")

    assert len(facts) > 0, "Should extract at least one fact"
    assert any('bob' in fact.lower() and 'chef' in fact.lower() for fact in facts), \
        "Should extract Bob's profession"
    assert not any('thank you' in fact.lower() for fact in facts), \
        "Should filter out pleasantries"


async def test_extract_facts_basic():
    """Test basic fact extraction."""

    text = "Alice works at Google. She loves Python programming."

    facts = await extract_facts(text)

    assert len(facts) > 0, "Should extract at least one fact"
    assert any('alice' in fact.lower() for fact in facts), "Should extract facts about Alice"


if __name__ == "__main__":
    import asyncio

    # Run a manual test
    async def main():
        conversation = """
        Host: Welcome to the AI podcast! Today we have Dr. Sarah Chen with us.
        Sarah: Hi! Thanks for having me.
        Host: So Sarah, tell us about your work.
        Sarah: I'm a researcher at Stanford focusing on large language models.
        Host: Oh wow!
        Sarah: Yeah, I've been studying how LLMs handle reasoning tasks. It's fascinating.
        Sarah: We published a paper last month showing that chain-of-thought prompting improves accuracy by 40%.
        Host: That's incredible!
        Sarah: And I'm also advising a startup called MemoryAI that's building long-term memory systems.
        Host: Cool, cool.
        """

        print("Testing fact extraction with podcast conversation:")
        print("=" * 60)
        facts = await extract_facts_from_text(conversation)
        print(f"\nExtracted {len(facts)} facts:\n")
        for i, fact in enumerate(facts, 1):
            print(f"{i}. {fact['fact']}")
            print(f"   Speaker: {fact['speaker']}, Type: {fact['type']}, Confidence: {fact['confidence']}\n")

    asyncio.run(main())
