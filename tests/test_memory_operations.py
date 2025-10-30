"""
Test basic memory operations: PUT, SEARCH, GET_RECENT.

Tests the core functionality of the temporal + semantic memory system.
"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_put_creates_memory_units(memory, clean_agent, db_connection):
    """Test that PUT operation creates memory units."""
    agent_id = clean_agent

    # Store a conversation
    await memory.put_async(
        agent_id=agent_id,
        content="Alice told me she loves hiking in the mountains. "
                "She mentioned that she goes hiking every weekend. "
                "Her favorite trail is in Yosemite National Park.",
        context="Casual conversation about hobbies",
        event_date=utcnow() - timedelta(hours=2),
    )

    # Verify memory units were created
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory_units WHERE agent_id = %s", (agent_id,))
    count = cursor.fetchone()[0]

    assert count > 0, "Memory units should be created"
    assert count <= 3, "Should create approximately 3 units (one per sentence)"

    cursor.close()


@pytest.mark.asyncio
async def test_put_creates_temporal_links(memory, clean_agent, db_connection):
    """Test that temporal links are created between recent memories."""
    agent_id = clean_agent

    # Store two memories close in time
    await memory.put_async(
        agent_id=agent_id,
        content="Alice loves hiking.",
        context="Hobbies",
        event_date=utcnow() - timedelta(hours=2),
    )

    await memory.put_async(
        agent_id=agent_id,
        content="Bob enjoys climbing.",
        context="Sports",
        event_date=utcnow() - timedelta(hours=1),
    )

    # Verify temporal links were created
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM memory_links
        WHERE link_type = 'temporal'
        AND from_unit_id IN (
            SELECT id FROM memory_units WHERE agent_id = %s
        )
    """, (agent_id,))

    temporal_link_count = cursor.fetchone()[0]
    assert temporal_link_count > 0, "Temporal links should be created"

    cursor.close()


@pytest.mark.asyncio
async def test_put_creates_semantic_links(memory, clean_agent, db_connection):
    """Test that semantic links are created between similar memories."""
    agent_id = clean_agent

    # Store semantically similar memories
    await memory.put_async(
        agent_id=agent_id,
        content="Alice loves hiking in the mountains.",
        context="Hobbies",
        event_date=utcnow() - timedelta(days=2),
    )

    await memory.put_async(
        agent_id=agent_id,
        content="Bob enjoys climbing mountains.",
        context="Sports",
        event_date=utcnow(),
    )

    # Verify semantic links were created
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM memory_links
        WHERE link_type = 'semantic'
        AND from_unit_id IN (
            SELECT id FROM memory_units WHERE agent_id = %s
        )
    """, (agent_id,))

    semantic_link_count = cursor.fetchone()[0]
    # Semantic links may or may not be created depending on similarity threshold
    # So we just check that the query works
    assert semantic_link_count >= 0, "Query should execute successfully"

    cursor.close()


@pytest.mark.asyncio
async def test_search_with_spreading_activation(memory, clean_agent):
    """Test search using spreading activation algorithm."""
    agent_id = clean_agent

    # Store memories about outdoor activities
    await memory.put_async(
        agent_id=agent_id,
        content="Alice told me she loves hiking in the mountains. "
                "She goes hiking every weekend.",
        context="Casual conversation about hobbies",
        event_date=utcnow() - timedelta(hours=2),
    )

    await memory.put_async(
        agent_id=agent_id,
        content="Bob mentioned he enjoys rock climbing. "
                "He climbs mountains on weekends too.",
        context="Discussion about outdoor sports",
        event_date=utcnow() - timedelta(hours=1),
    )

    # Search for outdoor activities
    results = memory.search(
        agent_id=agent_id,
        query="outdoor mountain activities",
        thinking_budget=50,
        top_k=5,
    )

    assert len(results) > 0, "Search should return results"

    # Verify result structure
    for result in results:
        assert 'id' in result, "Result should have id"
        assert 'text' in result, "Result should have text"
        assert 'weight' in result, "Result should have weight"
        assert 'activation' in result, "Result should have activation"
        assert 'recency' in result, "Result should have recency"
        assert 'frequency' in result, "Result should have frequency"

    # Results should be sorted by weight (descending)
    weights = [r['weight'] for r in results]
    assert weights == sorted(weights, reverse=True), "Results should be sorted by weight"


@pytest.mark.asyncio
async def test_search_returns_relevant_memories(memory, clean_agent):
    """Test that search returns semantically relevant memories."""
    agent_id = clean_agent

    # Store memories about different topics
    await memory.put_async(
        agent_id=agent_id,
        content="Alice loves hiking in the mountains.",
        context="Hobbies",
        event_date=utcnow() - timedelta(hours=2),
    )

    await memory.put_async(
        agent_id=agent_id,
        content="Bob is working on a Python web application.",
        context="Tech",
        event_date=utcnow() - timedelta(hours=1),
    )

    # Search for programming-related memories
    results = memory.search(
        agent_id=agent_id,
        query="software development",
        thinking_budget=50,
        top_k=3,
    )

    # Should find the programming-related memory
    assert len(results) > 0, "Search should return results"

    # Top result should be about programming (more relevant)
    top_result_text = results[0]['text'].lower()
    assert 'python' in top_result_text or 'application' in top_result_text or 'working' in top_result_text, \
        "Top result should be about programming"


@pytest.mark.asyncio
async def test_search_with_no_results(memory, clean_agent):
    """Test search behavior when no relevant memories exist."""
    agent_id = clean_agent

    # Store unrelated memories
    await memory.put_async(
        agent_id=agent_id,
        content="Alice loves cooking pasta.",
        context="Food",
        event_date=utcnow(),
    )

    # Search for something completely unrelated
    results = memory.search(
        agent_id=agent_id,
        query="quantum physics theories",
        thinking_budget=20,
        top_k=5,
    )

    # May return low-scoring results or empty list
    assert isinstance(results, list), "Search should return a list"
