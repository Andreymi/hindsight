#!/usr/bin/env python3
"""
Bank Transfer Tool - Export/Import banks without modifying hindsight-api.

Usage:
    python bank_transfer.py                           # Interactive mode
    python bank_transfer.py export <bank_id> -o <file.json>
    python bank_transfer.py import <bank_id> <file.json> [--mode merge-smart|merge|replace]
    python bank_transfer.py import-backup <bank_id> <backup.sql.gz> [--source-bank <bank>]
    python bank_transfer.py info <file.json>
"""

import argparse
import asyncio
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

try:
    import asyncpg
except ImportError:
    print("Error: asyncpg not installed. Run: pip install asyncpg", file=sys.stderr)
    sys.exit(1)

try:
    from pg0 import Pg0
except ImportError:
    Pg0 = None  # Optional - only needed if connecting to running pg0 instance


# --- Config ---

DEFAULT_USERNAME = "hindsight"
DEFAULT_PASSWORD = "hindsight"
DEFAULT_DATABASE = "hindsight"


def load_config() -> dict:
    """Load configuration from ~/.hindsight/embed or environment."""
    config = {
        "bank_id": os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default"),
        "db_url": os.environ.get("HINDSIGHT_API_DATABASE_URL"),
    }

    # Try to load from ~/.hindsight/embed
    config_path = Path.home() / ".hindsight" / "embed"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "HINDSIGHT_EMBED_BANK_ID":
                    config["bank_id"] = value
                elif key == "HINDSIGHT_API_DATABASE_URL":
                    config["db_url"] = value

    return config


async def get_pg0_uri(instance_name: str) -> str | None:
    """Get URI from a running pg0 instance."""
    if Pg0 is None:
        return None

    try:
        pg0 = Pg0(
            name=instance_name,
            username=DEFAULT_USERNAME,
            password=DEFAULT_PASSWORD,
            database=DEFAULT_DATABASE,
        )
        info = pg0.info()
        if info and info.running:
            return info.uri
    except Exception:
        pass
    return None


async def get_db_connection(config: dict, instance_name: str | None = None) -> asyncpg.Pool:
    """Get database connection pool."""
    # Try explicit DB URL first
    db_url = config.get("db_url")

    if not db_url and instance_name:
        # Try pg0
        db_url = await get_pg0_uri(instance_name)

    if not db_url:
        # Try default pg0 instance based on bank_id
        bank_id = config.get("bank_id", "default")
        pg0_instance = f"hindsight-embed-{bank_id}"
        db_url = await get_pg0_uri(pg0_instance)

    if not db_url:
        raise RuntimeError(
            "Could not determine database URL.\n"
            "Either:\n"
            "  1. Set HINDSIGHT_API_DATABASE_URL environment variable\n"
            "  2. Ensure hindsight-embed daemon is running (uvx hindsight-embed daemon start)\n"
            "  3. Use --db-url parameter"
        )

    return await asyncpg.create_pool(db_url, min_size=1, max_size=5)


# --- Serialization ---

def serialize_record(record: asyncpg.Record, include_embedding: bool = True) -> dict:
    """Convert asyncpg Record to JSON-serializable dict."""
    d = dict(record)
    for key, value in d.items():
        if isinstance(value, UUID):
            d[key] = str(value)
        elif isinstance(value, datetime):
            d[key] = value.isoformat()
        elif key == "embedding" and value is not None:
            if include_embedding:
                # pgvector returns numpy array or list - convert properly
                if hasattr(value, 'tolist'):
                    d[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    d[key] = list(value)
                elif isinstance(value, str):
                    # Already a string representation "[val1,val2,...]"
                    d[key] = value
                else:
                    d[key] = list(value)
            else:
                del d[key]
        elif isinstance(value, dict):
            pass  # JSON fields are already dicts
    return d


def parse_datetime(s: str | None) -> datetime | None:
    """Parse ISO datetime string or return None."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_uuid(s: str | None) -> UUID | None:
    """Parse UUID string or return None."""
    if not s:
        return None
    return UUID(s)


# --- Export ---

async def export_bank(pool: asyncpg.Pool, bank_id: str, include_embeddings: bool = True) -> dict:
    """Export all bank data to JSON-serializable dict."""
    async with pool.acquire() as conn:
        # 1. Bank profile
        bank = await conn.fetchrow(
            "SELECT * FROM banks WHERE bank_id = $1", bank_id
        )
        if not bank:
            raise ValueError(f"Bank '{bank_id}' not found")

        # 2. Documents
        documents = await conn.fetch(
            "SELECT id, original_text, content_hash, metadata, created_at "
            "FROM documents WHERE bank_id = $1",
            bank_id,
        )

        # 3. Chunks
        chunks = await conn.fetch(
            "SELECT chunk_id as id, document_id, chunk_text as text, chunk_index "
            "FROM chunks WHERE bank_id = $1",
            bank_id,
        )

        # 4. Memory units
        unit_cols = (
            "id, document_id, text, context, fact_type, "
            "occurred_start, occurred_end, mentioned_at, "
            "confidence_score, metadata, event_date"
        )
        if include_embeddings:
            unit_cols += ", embedding"
        units = await conn.fetch(
            f"SELECT {unit_cols} FROM memory_units WHERE bank_id = $1", bank_id
        )

        # 5. Entities
        entities = await conn.fetch(
            "SELECT id, canonical_name, metadata, first_seen, last_seen, mention_count "
            "FROM entities WHERE bank_id = $1",
            bank_id,
        )

        # 6. Unit-entity links
        unit_ids = [u["id"] for u in units]
        unit_entities = []
        if unit_ids:
            unit_entities = await conn.fetch(
                "SELECT unit_id, entity_id FROM unit_entities WHERE unit_id = ANY($1)",
                unit_ids,
            )

        # 7. Memory links
        memory_links = []
        if unit_ids:
            memory_links = await conn.fetch(
                "SELECT from_unit_id, to_unit_id, link_type, entity_id, weight "
                "FROM memory_links WHERE from_unit_id = ANY($1)",
                unit_ids,
            )

        # Determine embedding dimension
        embedding_dim = None
        if include_embeddings and units:
            first_emb = units[0].get("embedding")
            if first_emb:
                embedding_dim = len(first_emb)

        return {
            "meta": {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source_bank_id": bank_id,
                "include_embeddings": include_embeddings,
                "embedding_dimension": embedding_dim,
            },
            "bank": {
                "disposition": bank["disposition"] if bank["disposition"] else {},
                "background": bank["background"] or "",
            },
            "documents": [serialize_record(d) for d in documents],
            "chunks": [serialize_record(c) for c in chunks],
            "memory_units": [serialize_record(u, include_embeddings) for u in units],
            "entities": [serialize_record(e) for e in entities],
            "unit_entities": [serialize_record(ue) for ue in unit_entities],
            "memory_links": [serialize_record(ml) for ml in memory_links],
        }


# --- Import ---

def find_matching_entity(
    name: str, existing_map: dict[str, dict], threshold: float = 0.6
) -> dict | None:
    """Fuzzy match entity name against existing entities."""
    name_lower = name.lower()

    # Exact match
    if name_lower in existing_map:
        return existing_map[name_lower]

    # Fuzzy match
    best_score = 0.0
    best_match = None

    for existing_name, entity in existing_map.items():
        score = SequenceMatcher(None, name_lower, existing_name).ratio()
        if score > best_score:
            best_score = score
            best_match = entity

    if best_score >= threshold:
        return best_match
    return None


async def recalculate_cooccurrences(conn: asyncpg.Connection, bank_id: str) -> None:
    """Recalculate entity cooccurrences for bank."""
    # Delete existing
    await conn.execute(
        """
        DELETE FROM entity_cooccurrences
        WHERE entity_id_1 IN (SELECT id FROM entities WHERE bank_id = $1)
        """,
        bank_id,
    )

    # Rebuild from unit_entities
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        SELECT
            LEAST(ue1.entity_id, ue2.entity_id) as entity_id_1,
            GREATEST(ue1.entity_id, ue2.entity_id) as entity_id_2,
            COUNT(*) as cooccurrence_count,
            MAX(mu.mentioned_at) as last_cooccurred
        FROM unit_entities ue1
        JOIN unit_entities ue2 ON ue1.unit_id = ue2.unit_id AND ue1.entity_id < ue2.entity_id
        JOIN memory_units mu ON ue1.unit_id = mu.id
        WHERE mu.bank_id = $1
        GROUP BY LEAST(ue1.entity_id, ue2.entity_id), GREATEST(ue1.entity_id, ue2.entity_id)
        ON CONFLICT (entity_id_1, entity_id_2) DO UPDATE
        SET cooccurrence_count = EXCLUDED.cooccurrence_count,
            last_cooccurred = EXCLUDED.last_cooccurred
        """,
        bank_id,
    )


async def delete_bank_data(conn: asyncpg.Connection, bank_id: str) -> None:
    """Delete all data for a bank."""
    # Order matters due to foreign keys
    await conn.execute(
        "DELETE FROM memory_links WHERE from_unit_id IN (SELECT id FROM memory_units WHERE bank_id = $1)",
        bank_id,
    )
    await conn.execute(
        "DELETE FROM unit_entities WHERE unit_id IN (SELECT id FROM memory_units WHERE bank_id = $1)",
        bank_id,
    )
    await conn.execute(
        "DELETE FROM entity_cooccurrences WHERE entity_id_1 IN (SELECT id FROM entities WHERE bank_id = $1)",
        bank_id,
    )
    await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
    await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
    await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
    await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)


async def check_embedding_compatibility(
    pool: asyncpg.Pool, source_data: dict
) -> dict:
    """Check if source embeddings are compatible with target system."""
    # Get source dimension
    source_dim = source_data.get("meta", {}).get("embedding_dimension")
    if not source_dim and source_data.get("memory_units"):
        first_emb = source_data["memory_units"][0].get("embedding")
        if first_emb:
            source_dim = len(first_emb)

    # Get target dimension from existing data
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT embedding FROM memory_units LIMIT 1")
        target_dim = None
        if row and row["embedding"]:
            target_dim = len(row["embedding"])

    return {
        "source_dimension": source_dim,
        "target_dimension": target_dim,
        "compatible": source_dim == target_dim if source_dim and target_dim else True,
        "action": "use_as_is" if source_dim == target_dim else "regenerate_needed",
    }


async def analyze_import(
    conn: asyncpg.Connection, target_bank_id: str, data: dict, mode: str
) -> dict:
    """Dry-run analysis of import."""
    stats = {
        "documents_to_import": 0,
        "documents_to_skip": 0,
        "units_to_import": 0,
        "units_to_skip": 0,
        "entities_to_import": 0,
        "entities_to_merge": 0,
        "facts_by_type": {"world": 0, "opinion": 0, "observation": 0},
        "profile_comparison": {},
    }

    # Analyze facts by type
    for unit in data.get("memory_units", []):
        fact_type = unit.get("fact_type", "world")
        if fact_type in stats["facts_by_type"]:
            stats["facts_by_type"][fact_type] += 1

    # Compare profiles
    bank_data = data.get("bank", {})
    source_disposition = bank_data.get("disposition", {})
    source_background = bank_data.get("background", "")

    existing_bank = await conn.fetchrow(
        "SELECT disposition, background FROM banks WHERE bank_id = $1",
        target_bank_id,
    )

    if existing_bank:
        target_disposition = dict(existing_bank["disposition"]) if existing_bank["disposition"] else {}
        target_background = existing_bank["background"] or ""
        stats["profile_comparison"] = {
            "source_disposition": source_disposition,
            "target_disposition": target_disposition,
            "disposition_differs": source_disposition != target_disposition,
            "source_background_len": len(source_background),
            "target_background_len": len(target_background),
            "background_differs": source_background != target_background,
            "will_update": mode == "replace",
        }
    else:
        stats["profile_comparison"] = {
            "source_disposition": source_disposition,
            "target_disposition": None,
            "disposition_differs": True,
            "source_background_len": len(source_background),
            "target_background_len": 0,
            "background_differs": bool(source_background),
            "will_update": True,  # New bank, will use source profile
        }

    if mode == "replace":
        stats["documents_to_import"] = len(data.get("documents", []))
        stats["units_to_import"] = len(data.get("memory_units", []))
        stats["entities_to_import"] = len(data.get("entities", []))
        return stats

    # Check document deduplication
    for doc in data.get("documents", []):
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE bank_id = $1 AND content_hash = $2",
            target_bank_id,
            doc.get("content_hash"),
        )
        if existing:
            stats["documents_to_skip"] += 1
        else:
            stats["documents_to_import"] += 1

    # Simplified: count units based on documents
    skipped_doc_ids = set()
    for doc in data.get("documents", []):
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE bank_id = $1 AND content_hash = $2",
            target_bank_id,
            doc.get("content_hash"),
        )
        if existing:
            skipped_doc_ids.add(doc["id"])

    for unit in data.get("memory_units", []):
        if unit.get("document_id") in skipped_doc_ids:
            stats["units_to_skip"] += 1
        else:
            stats["units_to_import"] += 1

    # Check entity matching
    if mode == "merge-smart":
        existing_entities = await conn.fetch(
            "SELECT id, canonical_name FROM entities WHERE bank_id = $1",
            target_bank_id,
        )
        existing_map = {e["canonical_name"].lower(): e for e in existing_entities}

        for entity in data.get("entities", []):
            match = find_matching_entity(entity["canonical_name"], existing_map)
            if match:
                stats["entities_to_merge"] += 1
            else:
                stats["entities_to_import"] += 1
    else:
        stats["entities_to_import"] = len(data.get("entities", []))

    return stats


def show_profile_diff(source_disp: dict, target_disp: dict, source_bg: str, target_bg: str) -> None:
    """Display profile differences."""
    print("\n  Profile comparison:")
    print("  ┌─────────────┬──────────┬──────────┐")
    print("  │             │  Source  │  Target  │")
    print("  ├─────────────┼──────────┼──────────┤")
    for key in ["skepticism", "literalism", "empathy"]:
        src_val = source_disp.get(key, 3)
        tgt_val = target_disp.get(key, 3)
        marker = " *" if src_val != tgt_val else "  "
        print(f"  │ {key:<11} │    {src_val}     │    {tgt_val}    │{marker}")
    print("  └─────────────┴──────────┴──────────┘")
    if source_bg != target_bg:
        print(f"  Background: source={len(source_bg)} chars, target={len(target_bg)} chars *")
    print()


def ask_profile_action(source_disp: dict, target_disp: dict, source_bg: str, target_bg: str) -> str:
    """Ask user what to do with differing profiles."""
    show_profile_diff(source_disp, target_disp, source_bg, target_bg)
    print("  Profile differs. What to do?")
    print("    1. Keep target profile (default)")
    print("    2. Replace with source profile")
    print("    3. Abort import")
    choice = input("  Choice [1]: ").strip()
    if choice == "2":
        return "replace"
    elif choice == "3":
        return "abort"
    return "keep"


async def import_bank(
    pool: asyncpg.Pool,
    target_bank_id: str,
    data: dict,
    mode: str = "merge-smart",
    dry_run: bool = False,
    profile_action: str = "ask",  # "keep", "replace", "ask"
) -> dict:
    """Import bank data with UUID remapping."""
    stats = {
        "documents_imported": 0,
        "documents_skipped": 0,
        "chunks_imported": 0,
        "units_imported": 0,
        "units_skipped": 0,
        "entities_imported": 0,
        "entities_merged": 0,
        "links_imported": 0,
        "profile_updated": False,
    }

    # Get bank profile from export data
    bank_data = data.get("bank", {})
    source_disposition = bank_data.get("disposition", {"skepticism": 3, "literalism": 3, "empathy": 3})
    source_background = bank_data.get("background", "")

    async with pool.acquire() as conn:
        if dry_run:
            return await analyze_import(conn, target_bank_id, data, mode)

        # Check existing bank profile BEFORE transaction
        existing_bank = await conn.fetchrow(
            "SELECT disposition, background FROM banks WHERE bank_id = $1",
            target_bank_id,
        )

        # Determine profile update action
        update_profile = False
        if existing_bank is None:
            # New bank - always use source profile
            update_profile = True
        elif mode == "replace":
            # Replace mode - always overwrite
            update_profile = True
        else:
            # Merge modes - check if profiles differ
            target_disposition = dict(existing_bank["disposition"]) if existing_bank["disposition"] else {}
            target_background = existing_bank["background"] or ""

            profiles_differ = (
                source_disposition != target_disposition or
                source_background != target_background
            )

            if profiles_differ:
                if profile_action == "ask":
                    action = ask_profile_action(
                        source_disposition, target_disposition,
                        source_background, target_background
                    )
                    if action == "abort":
                        print("Import aborted by user.")
                        return stats
                    update_profile = (action == "replace")
                elif profile_action == "replace":
                    update_profile = True
                # else: profile_action == "keep" -> update_profile stays False

        async with conn.transaction():
            # 1. Handle mode
            if mode == "replace":
                await delete_bank_data(conn, target_bank_id)

            # 2. Ensure bank exists and update profile
            if existing_bank is None:
                # Create new bank with source profile
                await conn.execute(
                    """
                    INSERT INTO banks (bank_id, name, disposition, background)
                    VALUES ($1, $1, $2::jsonb, $3)
                    """,
                    target_bank_id,
                    json.dumps(source_disposition),
                    source_background,
                )
                stats["profile_updated"] = True
            elif update_profile:
                # Update existing bank with source profile
                await conn.execute(
                    """
                    UPDATE banks SET disposition = $2::jsonb, background = $3
                    WHERE bank_id = $1
                    """,
                    target_bank_id,
                    json.dumps(source_disposition),
                    source_background,
                )
                stats["profile_updated"] = True

            # 3. Build remap tables
            uuid_map: dict[str, UUID] = {}  # For entities, units (UUID columns)
            doc_uuid_map: dict[str, str] = {}  # For documents (TEXT columns)
            skipped_docs: set[str] = set()

            # 4. Import documents (dedupe by content_hash)
            # Note: documents.id is TEXT (not UUID)
            for doc in data.get("documents", []):
                doc_id = str(doc["id"])  # Keep as string
                if mode in ("merge", "merge-smart"):
                    existing = await conn.fetchrow(
                        "SELECT id FROM documents WHERE bank_id = $1 AND content_hash = $2",
                        target_bank_id,
                        doc.get("content_hash"),
                    )
                    if existing:
                        skipped_docs.add(doc_id)
                        doc_uuid_map[doc_id] = existing["id"]
                        stats["documents_skipped"] += 1
                        continue

                # Generate new text ID for document
                new_doc_id = str(uuid4())
                doc_uuid_map[doc_id] = new_doc_id

                metadata = doc.get("metadata")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                await conn.execute(
                    """
                    INSERT INTO documents (id, bank_id, original_text, content_hash, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                    """,
                    new_doc_id,
                    target_bank_id,
                    doc.get("original_text"),
                    doc.get("content_hash"),
                    json.dumps(metadata) if metadata else "{}",
                    parse_datetime(doc.get("created_at")) or datetime.now(timezone.utc),
                )
                stats["documents_imported"] += 1

            # 5. Import chunks (skip if document skipped)
            for chunk in data.get("chunks", []):
                chunk_doc_id = str(chunk.get("document_id")) if chunk.get("document_id") else None
                if chunk_doc_id and chunk_doc_id in skipped_docs:
                    continue

                mapped_doc_id = doc_uuid_map.get(chunk_doc_id) if chunk_doc_id else None
                chunk_index = chunk.get("chunk_index", 0)
                # chunk_id format: bank_id_document_id_chunk_index
                chunk_id = chunk.get("id") or f"{target_bank_id}_{mapped_doc_id}_{chunk_index}"

                await conn.execute(
                    """
                    INSERT INTO chunks (chunk_id, bank_id, document_id, chunk_text, chunk_index)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT DO NOTHING
                    """,
                    chunk_id,
                    target_bank_id,
                    mapped_doc_id,
                    chunk["text"],
                    chunk_index,
                )
                stats["chunks_imported"] += 1

            # 6. Import memory_units
            for unit in data.get("memory_units", []):
                unit_id = str(unit["id"])
                unit_doc_id = str(unit.get("document_id")) if unit.get("document_id") else None
                if unit_doc_id and unit_doc_id in skipped_docs:
                    stats["units_skipped"] += 1
                    continue

                new_unit_id = uuid4()
                uuid_map[unit_id] = new_unit_id

                # Convert embedding to pgvector format string
                embedding = unit.get("embedding")
                if embedding:
                    if isinstance(embedding, str):
                        # Already in pgvector string format "[val1,val2,...]"
                        pass
                    elif isinstance(embedding, list):
                        # Handle nested lists (e.g., [[values]])
                        while isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                            embedding = embedding[0]
                        if isinstance(embedding, list):
                            # Check if it's a list of numbers or a list of chars (broken)
                            if embedding and isinstance(embedding[0], str) and len(embedding[0]) == 1:
                                # List of characters - join back to string
                                embedding = "".join(embedding)
                            else:
                                embedding = "[" + ",".join(str(x) for x in embedding) + "]"

                metadata = unit.get("metadata")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # document_id in memory_units is TEXT, not UUID
                mapped_doc_id = doc_uuid_map.get(unit_doc_id) if unit_doc_id else None

                # event_date is required, use mentioned_at or now as fallback
                event_date = (
                    parse_datetime(unit.get("event_date"))
                    or parse_datetime(unit.get("mentioned_at"))
                    or datetime.now(timezone.utc)
                )

                await conn.execute(
                    """
                    INSERT INTO memory_units (
                        id, bank_id, document_id, text, embedding, context,
                        fact_type, occurred_start, occurred_end, mentioned_at,
                        confidence_score, metadata, event_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    new_unit_id,
                    target_bank_id,
                    mapped_doc_id,
                    unit["text"],
                    embedding,
                    unit.get("context"),
                    unit.get("fact_type", "world"),
                    parse_datetime(unit.get("occurred_start")),
                    parse_datetime(unit.get("occurred_end")),
                    parse_datetime(unit.get("mentioned_at")),
                    unit.get("confidence_score"),
                    json.dumps(metadata) if metadata else "{}",
                    event_date,
                )
                stats["units_imported"] += 1

            # 7. Import entities
            if mode == "merge-smart":
                existing_entities = await conn.fetch(
                    "SELECT id, canonical_name, mention_count FROM entities WHERE bank_id = $1",
                    target_bank_id,
                )
                existing_map = {
                    e["canonical_name"].lower(): dict(e) for e in existing_entities
                }

                for entity in data.get("entities", []):
                    entity_id = entity["id"]
                    match = find_matching_entity(entity["canonical_name"], existing_map)
                    if match:
                        uuid_map[entity_id] = match["id"]
                        await conn.execute(
                            "UPDATE entities SET mention_count = mention_count + $1 WHERE id = $2",
                            entity.get("mention_count", 1),
                            match["id"],
                        )
                        stats["entities_merged"] += 1
                    else:
                        new_entity_id = uuid4()
                        uuid_map[entity_id] = new_entity_id

                        metadata = entity.get("metadata")
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)

                        await conn.execute(
                            """
                            INSERT INTO entities (id, bank_id, canonical_name, metadata, mention_count, first_seen, last_seen)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            new_entity_id,
                            target_bank_id,
                            entity["canonical_name"],
                            json.dumps(metadata) if metadata else "{}",
                            entity.get("mention_count", 1),
                            parse_datetime(entity.get("first_seen")),
                            parse_datetime(entity.get("last_seen")),
                        )
                        stats["entities_imported"] += 1
                        existing_map[entity["canonical_name"].lower()] = {
                            "id": new_entity_id,
                            "canonical_name": entity["canonical_name"],
                        }
            else:
                for entity in data.get("entities", []):
                    entity_id = entity["id"]
                    new_entity_id = uuid4()
                    uuid_map[entity_id] = new_entity_id

                    metadata = entity.get("metadata")
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    await conn.execute(
                        """
                        INSERT INTO entities (id, bank_id, canonical_name, metadata, mention_count, first_seen, last_seen)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        new_entity_id,
                        target_bank_id,
                        entity["canonical_name"],
                        json.dumps(metadata) if metadata else "{}",
                        entity.get("mention_count", 1),
                        parse_datetime(entity.get("first_seen")),
                        parse_datetime(entity.get("last_seen")),
                    )
                    stats["entities_imported"] += 1

            # 8. Import unit_entities
            for ue in data.get("unit_entities", []):
                new_unit_id = uuid_map.get(ue["unit_id"])
                new_entity_id = uuid_map.get(ue["entity_id"])
                if new_unit_id and new_entity_id:
                    await conn.execute(
                        "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        new_unit_id,
                        new_entity_id,
                    )

            # 9. Import memory_links
            for link in data.get("memory_links", []):
                new_from = uuid_map.get(link["from_unit_id"])
                new_to = uuid_map.get(link["to_unit_id"])
                new_entity = (
                    uuid_map.get(link["entity_id"]) if link.get("entity_id") else None
                )
                if new_from and new_to:
                    await conn.execute(
                        """
                        INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, entity_id, weight)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        new_from,
                        new_to,
                        link["link_type"],
                        new_entity,
                        link.get("weight", 1.0),
                    )
                    stats["links_imported"] += 1

            # 10. Recalculate entity cooccurrences
            await recalculate_cooccurrences(conn, target_bank_id)

    return stats


# --- SQL Backup Import ---

def parse_sql_dump(sql: str, bank_id: str | None = None) -> dict:
    """Parse PostgreSQL COPY statements and extract bank data."""
    result: dict[str, Any] = {
        "meta": {"version": "1.0", "source": "sql_backup"},
        "documents": [],
        "chunks": [],
        "memory_units": [],
        "entities": [],
        "unit_entities": [],
        "memory_links": [],
    }

    # Find all bank_ids if not specified
    all_bank_ids: set[str] = set()

    # Pattern for COPY statements
    copy_pattern = r"COPY\s+(\w+)\s*\(([^)]+)\)\s*FROM\s+stdin;(.*?)\\."

    for match in re.finditer(copy_pattern, sql, re.DOTALL | re.IGNORECASE):
        table_name = match.group(1).lower()
        columns = [c.strip().strip('"') for c in match.group(2).split(",")]
        data_block = match.group(3).strip()

        bank_id_idx = None
        if "bank_id" in columns:
            bank_id_idx = columns.index("bank_id")

        for line in data_block.split("\n"):
            if not line.strip() or line.strip() == "\\.":
                continue

            values = line.split("\t")
            if len(values) != len(columns):
                continue

            # Handle \N (NULL) values
            values = [None if v == "\\N" else v for v in values]
            row = dict(zip(columns, values))

            # Track all bank_ids
            if bank_id_idx is not None and values[bank_id_idx]:
                all_bank_ids.add(values[bank_id_idx])

            # Filter by bank_id if specified
            if bank_id and row.get("bank_id") != bank_id:
                continue

            if table_name == "documents":
                result["documents"].append(row)
            elif table_name == "chunks":
                result["chunks"].append(row)
            elif table_name == "memory_units":
                result["memory_units"].append(row)
            elif table_name == "entities":
                result["entities"].append(row)
            elif table_name == "unit_entities":
                result["unit_entities"].append(row)
            elif table_name == "memory_links":
                result["memory_links"].append(row)

    result["meta"]["all_bank_ids"] = sorted(all_bank_ids)
    return result


def list_banks_in_backup(backup_path: Path) -> list[dict]:
    """List all bank_ids found in backup file with stats."""
    if backup_path.suffix == ".gz":
        with gzip.open(backup_path, "rt", errors="replace") as f:
            sql_content = f.read()
    else:
        sql_content = backup_path.read_text(errors="replace")

    data = parse_sql_dump(sql_content)
    all_bank_ids = data["meta"].get("all_bank_ids", [])

    banks = []
    for bid in all_bank_ids:
        bank_data = parse_sql_dump(sql_content, bid)
        banks.append({
            "bank_id": bid,
            "documents": len(bank_data["documents"]),
            "facts": len(bank_data["memory_units"]),
            "entities": len(bank_data["entities"]),
        })

    return banks


async def import_from_sql_backup(
    pool: asyncpg.Pool,
    target_bank_id: str,
    backup_path: Path,
    source_bank_id: str | None = None,
    mode: str = "merge-smart",
    profile_action: str = "ask",
) -> dict:
    """Import bank from PostgreSQL SQL dump."""
    # Read and decompress
    if backup_path.suffix == ".gz":
        with gzip.open(backup_path, "rt", errors="replace") as f:
            sql_content = f.read()
    else:
        sql_content = backup_path.read_text(errors="replace")

    # Parse SQL dump
    data = parse_sql_dump(sql_content, source_bank_id)

    if not source_bank_id:
        # No source bank specified - check what's available
        all_banks = data["meta"].get("all_bank_ids", [])
        if not all_banks:
            raise ValueError("No banks found in backup file")
        if len(all_banks) == 1:
            source_bank_id = all_banks[0]
            data = parse_sql_dump(sql_content, source_bank_id)
        else:
            raise ValueError(
                f"Multiple banks found in backup: {all_banks}. "
                f"Please specify --source-bank"
            )

    return await import_bank(pool, target_bank_id, data, mode, profile_action=profile_action)


# --- Interactive Mode ---

def print_box(title: str, lines: list[str]) -> None:
    """Print a nice box with content."""
    width = max(len(title) + 4, max(len(line) for line in lines) + 4)
    print(f"╭─ {title} {'─' * (width - len(title) - 4)}╮")
    for line in lines:
        print(f"│ {line.ljust(width - 4)} │")
    print(f"╰{'─' * (width - 2)}╯")


async def interactive_mode(pool: asyncpg.Pool, config: dict) -> None:
    """Run interactive wizard."""
    print()
    print_box("Bank Transfer Tool", [
        "1. Export bank",
        "2. Import bank",
        "3. Import from SQL backup",
        "4. List banks",
        "5. Info about export file",
        "q. Quit",
    ])

    while True:
        choice = input("\n> ").strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            await interactive_export(pool, config)
        elif choice == "2":
            await interactive_import(pool, config)
        elif choice == "3":
            await interactive_import_backup(pool, config)
        elif choice == "4":
            await interactive_list_banks(pool)
        elif choice == "5":
            interactive_info()
        else:
            print("Invalid choice. Try again.")


async def interactive_export(pool: asyncpg.Pool, config: dict) -> None:
    """Interactive export wizard."""
    bank_id = input(f"Bank ID [{config.get('bank_id', 'default')}]: ").strip()
    if not bank_id:
        bank_id = config.get("bank_id", "default")

    output_file = input("Output file [backup.json]: ").strip()
    if not output_file:
        output_file = "backup.json"

    include_emb = input("Include embeddings? [Y/n]: ").strip().lower()
    include_embeddings = include_emb != "n"

    print(f"\nExporting bank '{bank_id}'...")
    try:
        data = await export_bank(pool, bank_id, include_embeddings)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✓ Exported to {output_file}")
        print(f"  Documents: {len(data['documents'])}")
        print(f"  Facts: {len(data['memory_units'])}")
        print(f"  Entities: {len(data['entities'])}")
    except Exception as e:
        print(f"✗ Export failed: {e}")


async def interactive_import(pool: asyncpg.Pool, config: dict) -> None:
    """Interactive import wizard."""
    source_file = input("Source file: ").strip()
    if not source_file or not Path(source_file).exists():
        print("File not found")
        return

    target_bank = input(f"Target bank [{config.get('bank_id', 'default')}]: ").strip()
    if not target_bank:
        target_bank = config.get("bank_id", "default")

    with open(source_file) as f:
        data = json.load(f)

    # Check compatibility
    compat = await check_embedding_compatibility(pool, data)
    meta = data.get("meta", {})

    print()
    print_box("Compatibility Check", [
        f"Source embedding dim: {compat['source_dimension'] or 'N/A'}",
        f"Target embedding dim: {compat['target_dimension'] or 'N/A'}  "
        + ("✓ Compatible" if compat["compatible"] else "⚠ May need re-embed"),
        "",
        f"Source bank: {meta.get('source_bank_id', 'unknown')}",
        f"Source facts: {len(data.get('memory_units', []))}",
        f"Source entities: {len(data.get('entities', []))}",
        f"Source documents: {len(data.get('documents', []))}",
    ])

    print("\nImport mode:")
    print("  1. merge-smart (recommended) - dedupe + entity matching")
    print("  2. merge - dedupe only, new entities")
    print("  3. replace - delete target, full import")

    mode_choice = input("\nSelect mode [1]: ").strip()
    mode_map = {"1": "merge-smart", "2": "merge", "3": "replace", "": "merge-smart"}
    mode = mode_map.get(mode_choice, "merge-smart")

    # Dry run first
    print("\nAnalyzing import...")
    stats = await import_bank(pool, target_bank, data, mode, dry_run=True)

    # Show facts breakdown
    facts_by_type = stats.get("facts_by_type", {})
    facts_lines = [
        f"Documents to import: {stats.get('documents_to_import', 0)}",
        f"Documents to skip: {stats.get('documents_to_skip', 0)}",
        f"Facts to import: {stats.get('units_to_import', 0)}",
        f"  - world: {facts_by_type.get('world', 0)}",
        f"  - opinions: {facts_by_type.get('opinion', 0)}",
        f"  - observations: {facts_by_type.get('observation', 0)}",
        f"Facts to skip: {stats.get('units_to_skip', 0)}",
        f"Entities to import: {stats.get('entities_to_import', 0)}",
        f"Entities to merge: {stats.get('entities_to_merge', 0)}",
    ]

    # Show profile comparison if differs
    profile_cmp = stats.get("profile_comparison", {})
    if profile_cmp.get("disposition_differs") or profile_cmp.get("background_differs"):
        facts_lines.append("")
        facts_lines.append("Profile differs! (will prompt during import)")

    print_box("Dry Run Results", facts_lines)

    proceed = input("\nProceed with import? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Import cancelled")
        return

    print("\nImporting...")
    stats = await import_bank(pool, target_bank, data, mode)
    print(f"✓ Import complete")
    print(f"  Documents: {stats['documents_imported']} imported, {stats['documents_skipped']} skipped")
    print(f"  Facts: {stats['units_imported']} imported, {stats['units_skipped']} skipped")
    print(f"  Entities: {stats['entities_imported']} imported, {stats['entities_merged']} merged")
    print(f"  Links: {stats['links_imported']} imported")


async def interactive_import_backup(pool: asyncpg.Pool, config: dict) -> None:
    """Interactive SQL backup import wizard."""
    backup_file = input("Backup file (.sql or .sql.gz): ").strip()
    if not backup_file or not Path(backup_file).exists():
        print("File not found")
        return

    print("\nScanning backup file...")
    banks = list_banks_in_backup(Path(backup_file))

    if not banks:
        print("No banks found in backup")
        return

    print("\nFound banks:")
    for i, bank in enumerate(banks, 1):
        print(f"  {i}. {bank['bank_id']} ({bank['facts']} facts, {bank['entities']} entities)")

    if len(banks) == 1:
        source_bank = banks[0]["bank_id"]
        print(f"\nUsing only available bank: {source_bank}")
    else:
        choice = input(f"\nSelect source bank [1-{len(banks)}]: ").strip()
        try:
            idx = int(choice) - 1
            source_bank = banks[idx]["bank_id"]
        except (ValueError, IndexError):
            print("Invalid choice")
            return

    target_bank = input(f"Target bank [{config.get('bank_id', 'default')}]: ").strip()
    if not target_bank:
        target_bank = config.get("bank_id", "default")

    print("\nImport mode:")
    print("  1. merge-smart (recommended)")
    print("  2. merge")
    print("  3. replace")

    mode_choice = input("\nSelect mode [1]: ").strip()
    mode_map = {"1": "merge-smart", "2": "merge", "3": "replace", "": "merge-smart"}
    mode = mode_map.get(mode_choice, "merge-smart")

    proceed = input(f"\nImport '{source_bank}' -> '{target_bank}' ({mode})? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Import cancelled")
        return

    print("\nImporting from backup...")
    stats = await import_from_sql_backup(pool, target_bank, Path(backup_file), source_bank, mode)
    print(f"✓ Import complete")
    for k, v in stats.items():
        print(f"  {k}: {v}")


async def interactive_list_banks(pool: asyncpg.Pool) -> None:
    """List all banks in database."""
    async with pool.acquire() as conn:
        banks = await conn.fetch(
            """
            SELECT
                b.bank_id,
                COUNT(DISTINCT d.id) as documents,
                COUNT(DISTINCT m.id) as facts,
                COUNT(DISTINCT e.id) as entities
            FROM banks b
            LEFT JOIN documents d ON d.bank_id = b.bank_id
            LEFT JOIN memory_units m ON m.bank_id = b.bank_id
            LEFT JOIN entities e ON e.bank_id = b.bank_id
            GROUP BY b.bank_id
            ORDER BY b.bank_id
            """
        )

    if not banks:
        print("No banks found")
        return

    print("\nBanks in database:")
    for bank in banks:
        print(f"  {bank['bank_id']}: {bank['facts']} facts, {bank['entities']} entities, {bank['documents']} documents")


def interactive_info() -> None:
    """Show info about export file."""
    file_path = input("Export file: ").strip()
    if not file_path or not Path(file_path).exists():
        print("File not found")
        return

    with open(file_path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    print(f"\nExport file: {file_path}")
    print(f"  Version: {meta.get('version')}")
    print(f"  Source bank: {meta.get('source_bank_id')}")
    print(f"  Exported at: {meta.get('exported_at')}")
    print(f"  Documents: {len(data.get('documents', []))}")
    print(f"  Facts: {len(data.get('memory_units', []))}")
    print(f"  Entities: {len(data.get('entities', []))}")
    print(f"  Embeddings: {'yes' if meta.get('include_embeddings') else 'no'}")
    if meta.get("embedding_dimension"):
        print(f"  Embedding dimension: {meta.get('embedding_dimension')}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Bank Transfer Tool - Export/Import hindsight banks"
    )
    parser.add_argument(
        "--db-url",
        help="PostgreSQL connection URL (default: auto-detect from pg0)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Export
    export_p = subparsers.add_parser("export", help="Export bank to JSON")
    export_p.add_argument("bank_id", help="Bank ID to export")
    export_p.add_argument("-o", "--output", required=True, help="Output file")
    export_p.add_argument(
        "--no-embeddings", action="store_true", help="Exclude embeddings"
    )

    # Import
    import_p = subparsers.add_parser("import", help="Import bank from JSON")
    import_p.add_argument("bank_id", help="Target bank ID")
    import_p.add_argument("file", help="JSON file to import")
    import_p.add_argument(
        "--mode",
        choices=["merge-smart", "merge", "replace"],
        default="merge-smart",
        help="Import mode (default: merge-smart)",
    )
    import_p.add_argument(
        "--dry-run", action="store_true", help="Analyze without importing"
    )
    import_p.add_argument(
        "--profile",
        choices=["ask", "keep", "replace"],
        default="ask",
        help="Profile conflict action: ask (interactive), keep target, replace with source (default: ask)",
    )

    # Import from SQL backup
    backup_p = subparsers.add_parser(
        "import-backup", help="Import bank from SQL backup"
    )
    backup_p.add_argument("bank_id", help="Target bank ID")
    backup_p.add_argument("file", help="SQL backup file (.sql or .sql.gz)")
    backup_p.add_argument("--source-bank", help="Source bank ID in backup")
    backup_p.add_argument(
        "--mode",
        choices=["merge-smart", "merge", "replace"],
        default="merge-smart",
        help="Import mode",
    )
    backup_p.add_argument(
        "--profile",
        choices=["ask", "keep", "replace"],
        default="ask",
        help="Profile conflict action: ask (interactive), keep target, replace with source (default: ask)",
    )

    # Info
    info_p = subparsers.add_parser("info", help="Show export file info")
    info_p.add_argument("file", help="JSON or SQL backup file")

    # List banks in backup
    list_p = subparsers.add_parser("list-backup", help="List banks in SQL backup")
    list_p.add_argument("file", help="SQL backup file")

    args = parser.parse_args()

    # Run async main
    asyncio.run(async_main(args))


async def async_main(args):
    config = load_config()

    # Override DB URL if provided
    if hasattr(args, "db_url") and args.db_url:
        config["db_url"] = args.db_url

    # Handle commands that don't need DB connection
    if args.command == "info":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {args.file}")
            sys.exit(1)

        if file_path.suffix in (".gz", ".sql"):
            # SQL backup
            banks = list_banks_in_backup(file_path)
            print(f"SQL backup: {args.file}")
            print(f"  Banks found: {len(banks)}")
            for bank in banks:
                print(f"    - {bank['bank_id']}: {bank['facts']} facts, {bank['entities']} entities")
        else:
            # JSON export
            with open(args.file) as f:
                data = json.load(f)
            meta = data.get("meta", {})
            bank_data = data.get("bank", {})

            # Count facts by type
            facts_by_type = {"world": 0, "opinion": 0, "observation": 0}
            for unit in data.get("memory_units", []):
                fact_type = unit.get("fact_type", "world")
                if fact_type in facts_by_type:
                    facts_by_type[fact_type] += 1

            print(f"Export file: {args.file}")
            print(f"  Version: {meta.get('version')}")
            print(f"  Source bank: {meta.get('source_bank_id')}")
            print(f"  Exported at: {meta.get('exported_at')}")
            print(f"  Documents: {len(data.get('documents', []))}")
            print(f"  Facts: {len(data.get('memory_units', []))}")
            print(f"    - world: {facts_by_type['world']}")
            print(f"    - opinions: {facts_by_type['opinion']}")
            print(f"    - observations: {facts_by_type['observation']}")
            print(f"  Entities: {len(data.get('entities', []))}")
            print(f"  Embeddings: {'yes' if meta.get('include_embeddings') else 'no'}")
            if meta.get("embedding_dimension"):
                print(f"  Embedding dim: {meta.get('embedding_dimension')}")
            # Show profile info
            if bank_data:
                disposition = bank_data.get("disposition", {})
                background = bank_data.get("background", "")
                print(f"  Disposition: skepticism={disposition.get('skepticism', 3)}, "
                      f"literalism={disposition.get('literalism', 3)}, "
                      f"empathy={disposition.get('empathy', 3)}")
                if background:
                    print(f"  Background: {len(background)} chars")
        return

    if args.command == "list-backup":
        banks = list_banks_in_backup(Path(args.file))
        print(f"Banks in {args.file}:")
        for bank in banks:
            print(f"  {bank['bank_id']}: {bank['facts']} facts, {bank['entities']} entities, {bank['documents']} documents")
        return

    # Commands that need DB connection
    try:
        pool = await get_db_connection(config)
    except Exception as e:
        print(f"Error connecting to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if not args.command:
            # Interactive mode
            await interactive_mode(pool, config)

        elif args.command == "export":
            data = await export_bank(pool, args.bank_id, not args.no_embeddings)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✓ Exported to {args.output}")
            print(f"  Documents: {len(data['documents'])}")
            print(f"  Facts: {len(data['memory_units'])}")
            print(f"  Entities: {len(data['entities'])}")

        elif args.command == "import":
            with open(args.file) as f:
                data = json.load(f)
            stats = await import_bank(
                pool, args.bank_id, data, args.mode, args.dry_run,
                profile_action=args.profile
            )
            if args.dry_run:
                print("DRY RUN - no changes made")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            else:
                print("✓ Import complete")
                for k, v in stats.items():
                    print(f"  {k}: {v}")

        elif args.command == "import-backup":
            stats = await import_from_sql_backup(
                pool,
                args.bank_id,
                Path(args.file),
                args.source_bank,
                args.mode,
                profile_action=args.profile,
            )
            print("✓ Import complete")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    finally:
        await pool.close()


if __name__ == "__main__":
    main()
