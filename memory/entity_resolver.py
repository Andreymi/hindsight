"""
Entity extraction and resolution for memory system.

Uses spaCy for entity extraction and implements resolution logic
to disambiguate entities across memory units.
"""
import spacy
from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher


# Load spaCy model (singleton)
_nlp = None


def get_nlp():
    """Get or load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> List[Dict[str, any]]:
    """
    Extract entities from text using spaCy.

    Args:
        text: Input text

    Returns:
        List of entities with text, type, and span info
    """
    nlp = get_nlp()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        # Filter to important entity types
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })

    return entities


class EntityResolver:
    """
    Resolves entities to canonical IDs with disambiguation.
    """

    def __init__(self, db_conn):
        """
        Initialize entity resolver.

        Args:
            db_conn: psycopg2 database connection
        """
        self.conn = db_conn

    def resolve_entity(
        self,
        agent_id: str,
        entity_text: str,
        entity_type: str,
        context: str,
        nearby_entities: List[Dict],
        unit_event_date,
    ) -> str:
        """
        Resolve an entity to a canonical entity ID.

        Args:
            agent_id: Agent ID (entities are scoped to agents)
            entity_text: Entity text ("Alice", "Google", etc.)
            entity_type: Entity type (PERSON, ORG, etc.)
            context: Context where entity appears
            nearby_entities: Other entities in the same unit
            unit_event_date: When this unit was created

        Returns:
            Entity ID (creates new entity if needed)
        """
        cursor = self.conn.cursor()

        try:
            # Find candidate entities with same type and similar name
            cursor.execute(
                """
                SELECT id, canonical_name, metadata, last_seen
                FROM entities
                WHERE agent_id = %s
                  AND entity_type = %s
                  AND (
                    canonical_name ILIKE %s
                    OR canonical_name ILIKE %s
                    OR %s ILIKE canonical_name || '%%'
                  )
                ORDER BY mention_count DESC
                """,
                (agent_id, entity_type, entity_text, f"%{entity_text}%", entity_text)
            )

            candidates = cursor.fetchall()

            if not candidates:
                # New entity - create it
                return self._create_entity(
                    cursor, agent_id, entity_text, entity_type, unit_event_date
                )

            # Score candidates based on:
            # 1. Name similarity
            # 2. Context overlap (TODO: could use embeddings)
            # 3. Co-occurring entities
            # 4. Temporal proximity

            best_candidate = None
            best_score = 0.0
            best_name_similarity = 0.0

            nearby_entity_set = {e['text'].lower() for e in nearby_entities if e['text'] != entity_text}

            for candidate_id, canonical_name, metadata, last_seen in candidates:
                score = 0.0

                # 1. Name similarity (0-1)
                name_similarity = SequenceMatcher(
                    None,
                    entity_text.lower(),
                    canonical_name.lower()
                ).ratio()
                score += name_similarity * 0.5

                # 2. Co-occurring entities (0-0.5)
                # Get entities that co-occurred with this candidate before
                # Use the materialized co-occurrence cache for fast lookup
                cursor.execute(
                    """
                    SELECT e.canonical_name, ec.cooccurrence_count
                    FROM entity_cooccurrences ec
                    JOIN entities e ON (
                        CASE
                            WHEN ec.entity_id_1 = %s THEN ec.entity_id_2
                            WHEN ec.entity_id_2 = %s THEN ec.entity_id_1
                        END = e.id
                    )
                    WHERE ec.entity_id_1 = %s OR ec.entity_id_2 = %s
                    """,
                    (candidate_id, candidate_id, candidate_id, candidate_id)
                )
                co_entities = {row[0].lower() for row in cursor.fetchall()}

                # Check overlap with nearby entities
                overlap = len(nearby_entity_set & co_entities)
                if nearby_entity_set:
                    co_entity_score = overlap / len(nearby_entity_set)
                    score += co_entity_score * 0.3

                # 3. Temporal proximity (0-0.2)
                if last_seen:
                    days_diff = abs((unit_event_date - last_seen).total_seconds() / 86400)
                    if days_diff < 7:  # Within a week
                        temporal_score = max(0, 1.0 - (days_diff / 7))
                        score += temporal_score * 0.2

                if score > best_score:
                    best_score = score
                    best_candidate = candidate_id
                    best_name_similarity = name_similarity

            # Threshold for considering it the same entity
            # For PERSON entities with exact name match, use lower threshold
            threshold = 0.4 if entity_type == 'PERSON' and best_name_similarity >= 0.95 else 0.6

            if best_score > threshold:
                # Update entity
                cursor.execute(
                    """
                    UPDATE entities
                    SET mention_count = mention_count + 1,
                        last_seen = %s
                    WHERE id = %s
                    """,
                    (unit_event_date, best_candidate)
                )
                return best_candidate
            else:
                # Not confident - create new entity
                return self._create_entity(
                    cursor, agent_id, entity_text, entity_type, unit_event_date
                )

        finally:
            cursor.close()

    def _create_entity(
        self,
        cursor,
        agent_id: str,
        entity_text: str,
        entity_type: str,
        event_date,
    ) -> str:
        """
        Create a new entity.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            entity_text: Entity text
            entity_type: Entity type
            event_date: When first seen

        Returns:
            Entity ID
        """
        cursor.execute(
            """
            INSERT INTO entities (agent_id, canonical_name, entity_type, first_seen, last_seen, mention_count)
            VALUES (%s, %s, %s, %s, %s, 1)
            RETURNING id
            """,
            (agent_id, entity_text, entity_type, event_date, event_date)
        )
        entity_id = cursor.fetchone()[0]
        return entity_id

    def link_unit_to_entity(self, unit_id: str, entity_id: str):
        """
        Link a memory unit to an entity.
        Also updates co-occurrence cache with other entities in the same unit.

        Args:
            unit_id: Memory unit ID
            entity_id: Entity ID
        """
        cursor = self.conn.cursor()
        try:
            # Insert unit-entity link
            cursor.execute(
                """
                INSERT INTO unit_entities (unit_id, entity_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (unit_id, entity_id)
            )

            # Update co-occurrence cache: find other entities in this unit
            cursor.execute(
                """
                SELECT entity_id
                FROM unit_entities
                WHERE unit_id = %s AND entity_id != %s
                """,
                (unit_id, entity_id)
            )

            other_entities = [row[0] for row in cursor.fetchall()]

            # Update co-occurrences for each pair
            for other_entity_id in other_entities:
                self._update_cooccurrence(cursor, entity_id, other_entity_id)

        finally:
            cursor.close()

    def _update_cooccurrence(self, cursor, entity_id_1: str, entity_id_2: str):
        """
        Update the co-occurrence cache for two entities.

        Uses CHECK constraint ordering (entity_id_1 < entity_id_2) to avoid duplicates.

        Args:
            cursor: Database cursor
            entity_id_1: First entity ID
            entity_id_2: Second entity ID
        """
        # Ensure consistent ordering (smaller UUID first)
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        cursor.execute(
            """
            INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
            VALUES (%s, %s, 1, NOW())
            ON CONFLICT (entity_id_1, entity_id_2)
            DO UPDATE SET
                cooccurrence_count = entity_cooccurrences.cooccurrence_count + 1,
                last_cooccurred = NOW()
            """,
            (entity_id_1, entity_id_2)
        )

    def get_units_by_entity(self, entity_id: str, limit: int = 100) -> List[str]:
        """
        Get all units that mention an entity.

        Args:
            entity_id: Entity ID
            limit: Max results

        Returns:
            List of unit IDs
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                SELECT unit_id
                FROM unit_entities
                WHERE entity_id = %s
                ORDER BY unit_id
                LIMIT %s
                """,
                (entity_id, limit)
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_entity_by_text(
        self,
        agent_id: str,
        entity_text: str,
        entity_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Find an entity by text (for query resolution).

        Args:
            agent_id: Agent ID
            entity_text: Entity text to search for
            entity_type: Optional entity type filter

        Returns:
            Entity ID if found, None otherwise
        """
        cursor = self.conn.cursor()
        try:
            if entity_type:
                cursor.execute(
                    """
                    SELECT id FROM entities
                    WHERE agent_id = %s
                      AND entity_type = %s
                      AND canonical_name ILIKE %s
                    ORDER BY mention_count DESC
                    LIMIT 1
                    """,
                    (agent_id, entity_type, entity_text)
                )
            else:
                cursor.execute(
                    """
                    SELECT id FROM entities
                    WHERE agent_id = %s
                      AND canonical_name ILIKE %s
                    ORDER BY mention_count DESC
                    LIMIT 1
                    """,
                    (agent_id, entity_text)
                )

            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            cursor.close()
