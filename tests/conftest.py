"""
Pytest configuration and shared fixtures.
"""
import pytest
import psycopg2
import os
from dotenv import load_dotenv
from memory import TemporalSemanticMemory

load_dotenv()


@pytest.fixture(scope="function")
def memory():
    """
    Provide a clean memory system instance for each test.
    """
    mem = TemporalSemanticMemory()
    yield mem
    # Cleanup is handled by individual tests


@pytest.fixture(scope="function")
def clean_agent(memory):
    """
    Provide a clean agent ID and clean up data after test.
    """
    agent_id = "test_agent"

    # Clean up before test
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_units WHERE agent_id = %s", (agent_id,))
    cursor.execute("DELETE FROM entities WHERE agent_id = %s", (agent_id,))
    conn.commit()
    cursor.close()
    conn.close()

    yield agent_id

    # Clean up after test
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_units WHERE agent_id = %s", (agent_id,))
    cursor.execute("DELETE FROM entities WHERE agent_id = %s", (agent_id,))
    conn.commit()
    cursor.close()
    conn.close()


@pytest.fixture
def db_connection():
    """
    Provide a database connection for direct DB queries in tests.
    """
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    yield conn
    conn.close()
