"""
Test deduplication of identical puts.
"""
import pytest
from datetime import datetime, timezone
from memory.temporal_semantic_memory import TemporalSemanticMemory


@pytest.fixture
def memory():
    """Create a memory instance for testing."""
    mem = TemporalSemanticMemory()
    yield mem
    # Cleanup after test
    cursor = mem.conn.cursor()
    cursor.execute("DELETE FROM memory_units WHERE agent_id LIKE 'test_%'")
    mem.conn.commit()
    cursor.close()


@pytest.mark.asyncio
async def test_duplicate_put_filters_identical_content(memory):
    """Test that putting the same content twice doesn't create duplicates."""

    agent_id = "test_dedup_agent"
    content = "Alice works at Google as a software engineer. She joined last year and loves Python."
    event_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    # First put - should create units
    print("\n--- FIRST PUT ---")
    units_1 = await memory.put_async(agent_id, content, "Test context", event_date)

    assert len(units_1) > 0, "First put should create units"
    print(f"First put created {len(units_1)} units")

    # Second put with identical content and same date - should be filtered as duplicates
    print("\n--- SECOND PUT (identical) ---")
    units_2 = await memory.put_async(agent_id, content, "Test context", event_date)

    assert len(units_2) == 0, "Second identical put should create no new units (all duplicates)"
    print(f"Second put created {len(units_2)} units (expected 0)")

    # Verify database has only the first set of units
    cursor = memory.conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM memory_units WHERE agent_id = %s",
        (agent_id,)
    )
    total_units = cursor.fetchone()[0]
    cursor.close()

    assert total_units == len(units_1), f"Database should have {len(units_1)} units, found {total_units}"
    print(f"✅ Deduplication working: {total_units} total units in database")


@pytest.mark.asyncio
async def test_duplicate_put_with_paraphrased_content(memory):
    """Test that similar but paraphrased content is also deduplicated."""

    agent_id = "test_paraphrase_agent"
    event_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    # First put
    content_1 = "Bob is a chef in New York. He owns a restaurant."
    print("\n--- FIRST PUT ---")
    units_1 = await memory.put_async(agent_id, content_1, "Test", event_date)

    assert len(units_1) > 0, "First put should create units"
    print(f"First put created {len(units_1)} units")

    # Second put with paraphrased content - should be mostly deduplicated
    # The LLM will extract similar facts that should match via embeddings
    content_2 = "Bob works as a chef in New York City. He is the owner of a restaurant."
    print("\n--- SECOND PUT (paraphrased) ---")
    units_2 = await memory.put_async(agent_id, content_2, "Test", event_date)

    # May create 0 or very few new units (depending on how LLM extracts facts)
    print(f"Second put created {len(units_2)} units")
    print(f"Deduplication ratio: {len(units_2)}/{len(units_1)} new units from paraphrase")

    # Just verify it doesn't create the same number of units (some deduplication should happen)
    assert len(units_2) < len(units_1), "Paraphrased content should have fewer new units due to deduplication"


@pytest.mark.asyncio
async def test_different_dates_not_deduplicated(memory):
    """Test that same content with different dates is NOT deduplicated."""

    agent_id = "test_dates_agent"
    content = "Charlie went hiking in Yosemite."

    # First put at date 1
    date_1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    print("\n--- FIRST PUT (Jan 1) ---")
    units_1 = await memory.put_async(agent_id, content, "Test", date_1)

    assert len(units_1) > 0
    print(f"First put created {len(units_1)} units")

    # Second put at date 2 (outside 24-hour window)
    date_2 = datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    print("\n--- SECOND PUT (Feb 1, outside time window) ---")
    units_2 = await memory.put_async(agent_id, content, "Test", date_2)

    # Should create new units because dates are far apart
    assert len(units_2) > 0, "Same content with different dates (outside window) should create new units"
    print(f"Second put created {len(units_2)} units (not deduplicated due to date difference)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
