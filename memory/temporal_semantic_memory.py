"""
Temporal + Semantic + Entity Memory System for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio
import time

from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver, extract_entities
from .coref_resolver import resolve_sentences


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class TemporalSemanticMemory:
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the temporal + semantic memory system.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)
            embedding_model: Name of the SentenceTransformer model to use
        """
        load_dotenv()

        # Initialize PostgreSQL connection
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL not found. "
                "Set DATABASE_URL environment variable."
            )

        self.conn = psycopg2.connect(self.db_url)
        register_vector(self.conn)

        # Initialize entity resolver
        self.entity_resolver = EntityResolver(self.conn)

        # Initialize local embedding model (384 dimensions)
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Model loaded (embedding dim: {self.embedding_model.get_sentence_embedding_dimension()})")

    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using local SentenceTransformer model.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector (bge-small-en-v1.5)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model (batch processing).

        Local models are fast and process batches efficiently without needing
        parallel API calls. We run this in asyncio to avoid blocking, but the
        actual embedding generation is synchronous.

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embeddings in same order as input texts
        """
        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    def _find_duplicate_facts_batch(
        self,
        cursor,
        agent_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95
    ) -> List[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            cursor: Database cursor
            agent_id: Agent identifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        is_duplicate = []

        time_lower = event_date - timedelta(hours=time_window_hours)
        time_upper = event_date + timedelta(hours=time_window_hours)

        for text, embedding in zip(texts, embeddings):
            # Query for similar facts within time window
            cursor.execute(
                """
                SELECT id, text, 1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND event_date BETWEEN %s AND %s
                  AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT 1
                """,
                (embedding, agent_id, time_lower, time_upper, embedding, similarity_threshold)
            )

            result = cursor.fetchone()
            if result:
                is_duplicate.append(True)
            else:
                is_duplicate.append(False)

        return is_duplicate

    def put(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around put_async() for convenience.
        For best performance, use put_async() directly.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        return asyncio.run(self.put_async(agent_id, content, context, event_date))

    async def put_async(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This async version generates ALL embeddings in parallel for maximum speed,
        then uses batch inserts for database operations.

        Steps:
        1. Split content into sentence units
        2. Resolve coreferences
        3. **Generate ALL embeddings in parallel** (FAST!)
        4. **Batch insert all units and links** (FAST!)

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"PUT_ASYNC START: {agent_id}")
        print(f"Content length: {len(content)} chars")
        print(f"{'='*60}")

        if event_date is None:
            event_date = utcnow()

        # Step 1: Extract semantic facts using LLM (async)
        step_start = time.time()
        try:
            facts = await extract_facts(content)
            print(f"[1] Extract facts: {len(facts)} facts in {time.time() - step_start:.3f}s")
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"PUT_ASYNC FAILED: Fact extraction error")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            raise Exception(f"Failed to extract facts from content: {e}")

        # Step 2: Resolve pronouns to make facts even more self-contained
        step_start = time.time()
        sentences = resolve_sentences(facts)
        print(f"[2] Resolve coreferences: {time.time() - step_start:.3f}s")

        # Step 3: Generate ALL embeddings in parallel
        step_start = time.time()
        embeddings = await self._generate_embeddings_batch(sentences)
        print(f"[3] Generate embeddings (parallel): {len(embeddings)} embeddings in {time.time() - step_start:.3f}s")

        # Step 4: Check for duplicates using similarity + temporal window
        cursor = self.conn.cursor()
        step_start = time.time()
        duplicate_flags = self._find_duplicate_facts_batch(
            cursor, agent_id, sentences, embeddings, event_date
        )
        num_duplicates = sum(duplicate_flags)

        # Filter out duplicates
        filtered_data = [
            (sentence, embedding)
            for sentence, embedding, is_dup in zip(sentences, embeddings, duplicate_flags)
            if not is_dup
        ]

        if filtered_data:
            sentences, embeddings = zip(*filtered_data)
            sentences = list(sentences)
            embeddings = list(embeddings)
        else:
            sentences = []
            embeddings = []

        print(f"[4] Deduplication check: {num_duplicates} duplicates filtered, {len(sentences)} new facts in {time.time() - step_start:.3f}s")

        # If all facts were duplicates, return empty list
        if not sentences:
            cursor.close()
            print(f"\n{'='*60}")
            print(f"PUT_ASYNC COMPLETE: All facts were duplicates, nothing stored")
            print(f"{'='*60}\n")
            return []

        # Step 5: Batch insert everything
        try:
            # Batch INSERT all memory units
            step_start = time.time()
            from psycopg2.extras import execute_values
            unit_data = [
                (agent_id, sentence, embedding, context, event_date, 0)
                for sentence, embedding in zip(sentences, embeddings)
            ]

            unit_ids = execute_values(
                cursor,
                """
                INSERT INTO memory_units (agent_id, text, embedding, context, event_date, access_count)
                VALUES %s
                RETURNING id
                """,
                unit_data,
                fetch=True
            )
            created_unit_ids = [str(row[0]) for row in unit_ids]
            print(f"[5] Batch insert units: {time.time() - step_start:.3f}s")

            # Process entities for all units
            step_start = time.time()
            all_entity_links = []
            for unit_id, sentence in zip(created_unit_ids, sentences):
                entity_links = self._extract_entities_for_batch(cursor, agent_id, unit_id, sentence, context, event_date, sentences)
                all_entity_links.extend(entity_links)
            print(f"[6] Extract entities: {time.time() - step_start:.3f}s")

            # Create ALL temporal links in batch
            step_start = time.time()
            self._create_temporal_links_batch(cursor, agent_id, created_unit_ids, event_date)
            print(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

            # Create ALL semantic links in batch
            step_start = time.time()
            self._create_semantic_links_batch(cursor, agent_id, created_unit_ids, embeddings)
            print(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

            # Insert all entity links in batch
            step_start = time.time()
            if all_entity_links:
                self._insert_entity_links_batch(cursor, all_entity_links)
            print(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

            commit_start = time.time()
            self.conn.commit()
            print(f"[10] Commit: {time.time() - commit_start:.3f}s")

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"PUT_ASYNC COMPLETE: {len(created_unit_ids)} units stored in {total_time:.3f}s")
            print(f"{'='*60}\n")

            return created_unit_ids

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to store memory: {str(e)}")
        finally:
            cursor.close()

    def _create_temporal_links(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        event_date: datetime,
        time_window_hours: int = 24,
    ):
        """
        Create temporal links to recent memories.

        Links this unit to other units that occurred within a time window.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            unit_id: ID of the current unit
            event_date: When this event occurred
            time_window_hours: Size of the temporal window
        """
        try:
            # Get recent units within time window
            cursor.execute(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE agent_id = %s
                  AND id != %s
                  AND event_date >= %s
                ORDER BY event_date DESC
                LIMIT 10
                """,
                (agent_id, unit_id, event_date - timedelta(hours=time_window_hours))
            )

            recent_units = cursor.fetchall()

            # Create links to recent units
            links = []
            for recent_id, recent_event_date in recent_units:
                # Calculate temporal proximity weight
                time_diff_hours = abs((event_date - recent_event_date).total_seconds() / 3600)
                weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))

                links.append((unit_id, recent_id, 'temporal', weight, None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"Warning: Failed to create temporal links: {str(e)}")

    def _create_semantic_links(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links to similar memories.

        Links this unit to other units with similar meaning.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            unit_id: ID of the current unit
            embedding: Embedding of the current unit
            top_k: Number of similar units to link to
            threshold: Minimum similarity threshold
        """
        try:
            # Find similar units using vector similarity
            cursor.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND id != %s
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, agent_id, unit_id, embedding, threshold, embedding, top_k)
            )

            similar_units = cursor.fetchall()

            # Create links to similar units
            links = []
            for similar_id, similarity in similar_units:
                links.append((unit_id, similar_id, 'semantic', float(similarity), None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"Warning: Failed to create semantic links: {str(e)}")

    def _extract_and_link_entities(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        text: str,
        context: str,
        event_date,
        all_sentences: List[str],
    ):
        """
        Extract entities from text, resolve them, and create entity links.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            unit_id: Current unit ID
            text: Unit text
            context: Context
            event_date: When created
            all_sentences: All sentences from the same PUT (for context)
        """
        try:
            # Extract entities from this unit
            entities = extract_entities(text)

            if not entities:
                return

            # Resolve each entity and link
            entity_ids = []
            for entity in entities:
                entity_id = self.entity_resolver.resolve_entity(
                    agent_id=agent_id,
                    entity_text=entity['text'],
                    entity_type=entity['type'],
                    context=context,
                    nearby_entities=entities,
                    unit_event_date=event_date
                )
                entity_ids.append(entity_id)

                # Link unit to entity
                self.entity_resolver.link_unit_to_entity(unit_id, entity_id)

            # Create entity links to other units that mention the same entities
            for entity_id in set(entity_ids):
                # Get other units that mention this entity
                related_units = self.entity_resolver.get_units_by_entity(entity_id, limit=50)

                # Create entity links
                links = []
                for related_unit_id in related_units:
                    if str(related_unit_id) != str(unit_id):
                        links.append((unit_id, related_unit_id, 'entity', 1.0, entity_id))

                if links:
                    execute_values(
                        cursor,
                        """
                        INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                        """,
                        links
                    )

        except Exception as e:
            print(f"Warning: Failed to extract/link entities: {str(e)}")

    def search(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        live_tracer=None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using spreading activation.

        This implements the core SEARCH operation:
        1. Find entry points (most relevant units via vector search)
        2. Spread activation through the graph
        3. Weight results by activation + recency + frequency
        4. Return top results

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Step 1: Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Step 2: Find entry points
            cursor.execute(
                """
                SELECT id, text, context, event_date, access_count,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector)) >= 0.5
                ORDER BY embedding <=> %s::vector
                LIMIT 3
                """,
                (query_embedding, agent_id, query_embedding, query_embedding)
            )

            entry_points = cursor.fetchall()
            if not entry_points:
                return []

            # Step 3: Spreading activation with budget
            visited = set()
            results = []
            budget_remaining = thinking_budget
            queue = [(dict(unit), 1.0, True) for unit in entry_points]  # (unit, activation, is_entry)

            while queue and budget_remaining > 0:
                current_unit, activation, is_entry_point = queue.pop(0)
                unit_id = str(current_unit["id"])

                if unit_id in visited:
                    continue

                visited.add(unit_id)
                budget_remaining -= 1

                # Increment access count
                cursor.execute(
                    "UPDATE memory_units SET access_count = access_count + 1 WHERE id = %s",
                    (unit_id,)
                )

                # Calculate combined weight
                event_date = current_unit["event_date"]
                days_since = (utcnow() - event_date).total_seconds() / 86400

                recency_weight = calculate_recency_weight(days_since)
                frequency_weight = calculate_frequency_weight(current_unit.get("access_count", 0))

                # Combined weight: activation * recency * frequency
                final_weight = activation * recency_weight * frequency_weight

                # Notify tracer
                if live_tracer:
                    live_tracer.visit_node(
                        node_id=unit_id,
                        text=current_unit["text"],
                        activation=activation,
                        recency=recency_weight,
                        frequency=frequency_weight,
                        weight=final_weight,
                        is_entry_point=is_entry_point,
                    )
                    import time
                    time.sleep(0.15)  # Slow down for visualization

                results.append({
                    "id": unit_id,
                    "text": current_unit["text"],
                    "context": current_unit.get("context", ""),
                    "event_date": event_date.isoformat(),
                    "weight": final_weight,
                    "activation": activation,
                    "recency": recency_weight,
                    "frequency": frequency_weight,
                })

                # Spread to neighbors
                cursor.execute(
                    """
                    SELECT ml.to_unit_id, ml.weight, mu.text, mu.context, mu.event_date, mu.access_count
                    FROM memory_links ml
                    JOIN memory_units mu ON ml.to_unit_id = mu.id
                    WHERE ml.from_unit_id = %s
                      AND ml.weight >= 0.1
                    ORDER BY ml.weight DESC
                    """,
                    (unit_id,)
                )

                neighbors = cursor.fetchall()
                for neighbor in neighbors:
                    neighbor_id = str(neighbor["to_unit_id"])
                    if neighbor_id not in visited:
                        link_weight = neighbor["weight"]
                        new_activation = activation * link_weight * 0.8  # 0.8 = decay factor

                        if new_activation > 0.1:
                            queue.append(({
                                "id": neighbor["to_unit_id"],
                                "text": neighbor["text"],
                                "context": neighbor.get("context", ""),
                                "event_date": neighbor["event_date"],
                                "access_count": neighbor["access_count"],
                            }, new_activation, False))  # Not an entry point

            self.conn.commit()

            # Step 4: Sort by final weight and return top results
            results.sort(key=lambda x: x["weight"], reverse=True)
            return results[:top_k]

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to search memories: {str(e)}")
        finally:
            cursor.close()

    def get_memory_graph_data(self, agent_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Get memory graph data for visualization.

        Args:
            agent_id: Optional agent ID (if None, returns all data)

        Returns:
            Tuple of (units, links) for visualization
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get all units (optionally filtered by agent)
            if agent_id:
                cursor.execute(
                    "SELECT id, text, context, event_date, access_count FROM memory_units WHERE agent_id = %s",
                    (agent_id,)
                )
            else:
                cursor.execute(
                    "SELECT id, text, context, event_date, access_count FROM memory_units"
                )
            units = [dict(row) for row in cursor.fetchall()]

            # Get all links (optionally filtered by agent)
            if agent_id:
                cursor.execute(
                    """
                    SELECT ml.from_unit_id, ml.to_unit_id, ml.link_type, ml.weight
                    FROM memory_links ml
                    JOIN memory_units mu1 ON ml.from_unit_id = mu1.id
                    JOIN memory_units mu2 ON ml.to_unit_id = mu2.id
                    WHERE mu1.agent_id = %s
                    """,
                    (agent_id,)
                )
            else:
                cursor.execute(
                    "SELECT from_unit_id, to_unit_id, link_type, weight FROM memory_links"
                )
            links = [dict(row) for row in cursor.fetchall()]

            return units, links

        except Exception as e:
            raise Exception(f"Failed to get memory graph data: {str(e)}")
        finally:
            cursor.close()

    def _extract_entities_for_batch(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        text: str,
        context: str,
        event_date,
        all_sentences: List[str],
    ) -> List[tuple]:
        """
        Extract entities and return entity links (doesn't insert yet).

        Returns list of tuples for batch insertion: (from_unit_id, to_unit_id, link_type, weight, entity_id)
        """
        from .entity_resolver import extract_entities

        try:
            # Extract entities from this unit
            entities = extract_entities(text)

            if not entities:
                return []

            # Resolve each entity
            entity_ids = []
            for entity in entities:
                entity_id = self.entity_resolver.resolve_entity(
                    agent_id=agent_id,
                    entity_text=entity['text'],
                    entity_type=entity['type'],
                    context=context,
                    nearby_entities=entities,
                    unit_event_date=event_date
                )
                entity_ids.append(entity_id)

                # Link unit to entity (this inserts into entity_units)
                self.entity_resolver.link_unit_to_entity(unit_id, entity_id)

            # Now collect entity links for batch insertion
            # After link_unit_to_entity has been called, entity_units should exist
            links = []
            for entity_id in set(entity_ids):
                # Find all other units with this entity (cursor must be fresh)
                try:
                    cursor.execute(
                        """
                        SELECT unit_id
                        FROM unit_entities
                        WHERE entity_id = %s AND unit_id != %s
                        """,
                        (entity_id, unit_id)
                    )

                    related_units = cursor.fetchall()
                    for (related_unit_id,) in related_units:
                        # Bidirectional links
                        links.append((unit_id, related_unit_id, 'entity', 1.0, entity_id))
                        links.append((related_unit_id, unit_id, 'entity', 1.0, entity_id))
                except Exception as query_error:
                    # If there's an error querying, just skip this entity
                    print(f"Warning: Failed to query entity_units for {entity_id}: {str(query_error)}")
                    continue

            return links

        except Exception as e:
            print(f"Warning: Failed to extract entities: {str(e)}")
            return []

    def _create_temporal_links_batch(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        event_date: datetime,
        time_window_hours: int = 24,
    ):
        """
        Create temporal links for multiple units in one batch query.

        Uses a single query to find all relevant temporal connections.
        """
        if not unit_ids:
            return

        try:
            from psycopg2.extras import execute_values

            # Get ALL recent units within time window (single query)
            # Cast string IDs to UUIDs for comparison
            cursor.execute(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE agent_id = %s
                  AND id::text != ALL(%s)
                  AND event_date >= %s
                ORDER BY event_date DESC
                """,
                (agent_id, unit_ids, event_date - timedelta(hours=time_window_hours))
            )

            recent_units = cursor.fetchall()

            # Create links from each new unit to all recent units
            links = []
            for unit_id in unit_ids:
                for recent_id, recent_event_date in recent_units:
                    # Calculate temporal proximity weight
                    time_diff_hours = abs((event_date - recent_event_date).total_seconds() / 3600)
                    weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                    links.append((unit_id, recent_id, 'temporal', weight, None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"Warning: Failed to create temporal links: {str(e)}")

    def _create_semantic_links_batch(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links for multiple units efficiently.

        For each unit, finds similar units and creates links.
        """
        if not unit_ids or not embeddings:
            return

        try:
            from psycopg2.extras import execute_values

            all_links = []

            for unit_id, embedding in zip(unit_ids, embeddings):
                # Find similar units using vector similarity
                cursor.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = %s
                      AND id != %s
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, agent_id, unit_id, embedding, threshold, embedding, top_k)
                )

                similar_units = cursor.fetchall()

                for similar_id, similarity in similar_units:
                    all_links.append((unit_id, similar_id, 'semantic', float(similarity), None))

            if all_links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    all_links
                )

        except Exception as e:
            print(f"Warning: Failed to create semantic links: {str(e)}")

    def _insert_entity_links_batch(self, cursor, links: List[tuple]):
        """Insert all entity links in a single batch."""
        if not links:
            return

        try:
            from psycopg2.extras import execute_values
            execute_values(
                cursor,
                """
                INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES %s
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                links
            )
        except Exception as e:
            print(f"Warning: Failed to insert entity links: {str(e)}")
