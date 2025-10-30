"""
Test visualization functionality.

Tests memory graph data retrieval (not actual rendering).
"""
import pytest
from datetime import datetime, timedelta, timezone


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def test_get_memory_graph_data(memory, clean_agent):
    """Test retrieval of memory graph data for visualization."""
    agent_id = clean_agent

    # Store some memories
    memory.put(
        agent_id=agent_id,
        content="Alice loves hiking in the mountains.",
        context="Hobbies",
        event_date=utcnow() - timedelta(hours=2),
    )

    memory.put(
        agent_id=agent_id,
        content="Bob enjoys rock climbing.",
        context="Sports",
        event_date=utcnow() - timedelta(hours=1),
    )

    memory.put(
        agent_id=agent_id,
        content="Alice is working on a Python project.",
        context="Tech",
        event_date=utcnow(),
    )

    # Get graph data
    units, links = memory.get_memory_graph_data(agent_id)

    assert isinstance(units, list), "Units should be a list"
    assert isinstance(links, list), "Links should be a list"
    assert len(units) > 0, "Should have memory units"

    # Verify unit structure
    for unit in units:
        assert 'id' in unit, "Unit should have id"
        assert 'text' in unit, "Unit should have text"
        assert 'context' in unit, "Unit should have context"
        assert 'event_date' in unit, "Unit should have event_date"
        assert 'access_count' in unit, "Unit should have access_count"

    # Links may or may not exist depending on similarity/proximity
    if len(links) > 0:
        # Verify link structure
        for link in links:
            assert 'from_unit_id' in link, "Link should have from_unit_id"
            assert 'to_unit_id' in link, "Link should have to_unit_id"
            assert 'link_type' in link, "Link should have link_type"
            assert 'weight' in link, "Link should have weight"
            assert link['link_type'] in ['temporal', 'semantic', 'entity'], \
                "Link type should be temporal, semantic, or entity"


def test_memory_graph_has_correct_agent_data(memory, clean_agent):
    """Test that graph data only includes data for the specified agent."""
    agent_id = clean_agent
    other_agent_id = "other_agent"

    # Store memories for test agent
    memory.put(
        agent_id=agent_id,
        content="Alice loves hiking.",
        context="Hobbies",
        event_date=utcnow(),
    )

    # Store memories for another agent
    memory.put(
        agent_id=other_agent_id,
        content="Charlie enjoys swimming.",
        context="Sports",
        event_date=utcnow(),
    )

    # Get graph data for test agent
    units, links = memory.get_memory_graph_data(agent_id)

    # Should only include test agent's data
    for unit in units:
        # Verify by checking text content (Alice should be present, Charlie should not)
        unit_text = unit['text']
        assert 'Charlie' not in unit_text, "Should not include other agent's memories"
