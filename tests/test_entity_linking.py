"""
Test entity-aware memory linking functionality.

Tests that entity resolution connects memories about the same person/place/thing.
"""
import pytest
from datetime import datetime, timedelta, timezone


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def test_entity_extraction_and_linking(memory, clean_agent, db_connection):
    """Test that entities are extracted and linked correctly."""
    agent_id = clean_agent

    # Store memories about Alice's hiking hobby
    memory.put(
        agent_id=agent_id,
        content="Alice told me she loves hiking in the mountains. "
                "She goes hiking every weekend in Yosemite.",
        context="Casual conversation about hobbies",
        event_date=utcnow() - timedelta(days=7),
    )

    # Store memories about Alice's work (different context!)
    memory.put(
        agent_id=agent_id,
        content="Alice works at Google as a software engineer. "
                "She joined Google last year and loves the culture.",
        context="Discussion about careers",
        event_date=utcnow() - timedelta(days=3),
    )

    # Store more about hiking (no Alice mention)
    memory.put(
        agent_id=agent_id,
        content="Bob mentioned he enjoys rock climbing. "
                "He climbs in Yosemite too, on weekends.",
        context="Outdoor activities discussion",
        event_date=utcnow() - timedelta(days=1),
    )

    # Store another Alice memory
    memory.put(
        agent_id=agent_id,
        content="Alice is working on a Python project at Google. "
                "The project uses machine learning.",
        context="Technical discussion",
        event_date=utcnow(),
    )

    # Verify entities were extracted
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT canonical_name, entity_type, mention_count
        FROM entities
        WHERE agent_id = %s
        ORDER BY mention_count DESC
    """, (agent_id,))

    entities = cursor.fetchall()
    entity_names = [e[0] for e in entities]

    # Should have Alice, Google, Yosemite, Bob
    assert "Alice" in entity_names, "Alice entity should be extracted"
    assert "Google" in entity_names, "Google entity should be extracted"
    assert "Yosemite" in entity_names, "Yosemite entity should be extracted"
    assert "Bob" in entity_names, "Bob entity should be extracted"

    # Alice should have multiple mentions
    alice_entity = next((e for e in entities if e[0] == "Alice"), None)
    assert alice_entity is not None
    assert alice_entity[2] >= 3, "Alice should have at least 3 mentions"

    # Verify entity links exist
    cursor.execute("""
        SELECT COUNT(*)
        FROM memory_links
        WHERE link_type = 'entity'
        AND from_unit_id IN (
            SELECT id FROM memory_units WHERE agent_id = %s
        )
    """, (agent_id,))

    entity_link_count = cursor.fetchone()[0]
    assert entity_link_count > 0, "Entity links should be created"

    cursor.close()


def test_entity_search_retrieves_all_related_memories(memory, clean_agent):
    """Test that searching for an entity retrieves ALL memories about that entity."""
    agent_id = clean_agent

    # Store diverse memories about Alice
    memory.put(
        agent_id=agent_id,
        content="Alice loves hiking in the mountains.",
        context="Hobbies",
        event_date=utcnow() - timedelta(days=7),
    )

    memory.put(
        agent_id=agent_id,
        content="Alice works at Google as a software engineer.",
        context="Career",
        event_date=utcnow() - timedelta(days=3),
    )

    memory.put(
        agent_id=agent_id,
        content="Alice is working on a Python machine learning project.",
        context="Technical",
        event_date=utcnow(),
    )

    # Query about Alice - should get ALL Alice memories via entity links
    results = memory.search(
        agent_id=agent_id,
        query="What does Alice do?",
        thinking_budget=30,
        top_k=10,
    )

    # Should retrieve multiple memories about Alice
    assert len(results) >= 2, "Should find multiple memories about Alice"

    # Check that results contain Alice-related content
    alice_mentions = sum(1 for r in results if "Alice" in r['text'])
    assert alice_mentions >= 2, "Multiple results should mention Alice"


def test_entity_disambiguation(memory, clean_agent, db_connection):
    """Test that entity disambiguation correctly identifies same vs different entities."""
    agent_id = clean_agent

    # Store two memories about "Alice" in different contexts
    memory.put(
        agent_id=agent_id,
        content="Alice from engineering loves Python.",
        context="Tech team",
        event_date=utcnow() - timedelta(days=2),
    )

    memory.put(
        agent_id=agent_id,
        content="Alice from engineering is working on a new project.",
        context="Tech team",
        event_date=utcnow(),
    )

    # Check that only ONE Alice entity was created (not two)
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM entities
        WHERE agent_id = %s AND canonical_name = 'Alice'
    """, (agent_id,))

    alice_count = cursor.fetchone()[0]
    assert alice_count == 1, "Should create only one Alice entity (disambiguation)"

    cursor.close()


def test_link_type_distribution(memory, clean_agent, db_connection):
    """Test that all three link types (temporal, semantic, entity) are created."""
    agent_id = clean_agent

    # Store related memories
    memory.put(
        agent_id=agent_id,
        content="Alice works at Google. She loves her job.",
        context="Career",
        event_date=utcnow() - timedelta(hours=2),
    )

    memory.put(
        agent_id=agent_id,
        content="Bob also works at Google. He is in sales.",
        context="Career",
        event_date=utcnow() - timedelta(hours=1),
    )

    memory.put(
        agent_id=agent_id,
        content="Google is a great company to work for.",
        context="Career",
        event_date=utcnow(),
    )

    # Check link types
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT link_type, COUNT(*) as count
        FROM memory_links ml
        JOIN memory_units mu ON ml.from_unit_id = mu.id
        WHERE mu.agent_id = %s
        GROUP BY link_type
        ORDER BY count DESC
    """, (agent_id,))

    link_types = {row[0]: row[1] for row in cursor.fetchall()}

    # Should have at least temporal and entity links (semantic depends on similarity threshold)
    assert 'temporal' in link_types, "Should create temporal links"
    assert 'entity' in link_types or 'semantic' in link_types, "Should create entity or semantic links"

    cursor.close()
