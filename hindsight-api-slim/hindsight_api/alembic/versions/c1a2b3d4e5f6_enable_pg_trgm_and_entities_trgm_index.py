"""Enable pg_trgm extension and add GIN trigram index on entities.canonical_name

Revision ID: c1a2b3d4e5f6
Revises: b4c5d6e7f8a9
Create Date: 2026-03-02

Index is created CONCURRENTLY so the migration does not block writes on entities
during production deployments. CONCURRENTLY requires running outside a transaction
block; see migrations.py for how this is handled safely.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "c1a2b3d4e5f6"
down_revision: str | Sequence[str] | None = "b4c5d6e7f8a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    # pg_trgm ships with most PostgreSQL installations as a contrib module.
    # It enables fast similarity lookups via GIN indexes, used for entity name matching.
    # On managed services (e.g. Azure Flexible Server), the extension may not be
    # available or may require manual enablement.  We gracefully skip the index
    # creation if the extension cannot be loaded — the entity resolver will
    # auto-detect and fall back to the "full" lookup strategy at runtime.  See #626.
    # Pin pg_trgm to `extensions` so the % operator and similarity functions are
    # visible from any session's default search_path ('"$user", public, extensions'),
    # while keeping `public` empty per sup ADR-019 (schema-isolation: domain tables
    # in app, extensions in `extensions`, public stays empty).
    # In multi-tenant deployments (SupTenantExtension), this migration runs with
    # search_path TO "<tenant>", public — without an explicit WITH SCHEMA pg_trgm
    # would land in the tenant schema and be invisible to background workers using
    # the default postgres search_path.
    # See: PATCHED — `pg-trgm-extensions-schema` semgrep rule and corrective
    # migration `move_pg_trgm_to_extensions_per_adr019` for existing installs.
    conn = op.get_bind()
    try:
        conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS extensions"))
        # Idempotently grant USAGE to Supabase roles if they exist; pg0 (single-schema
        # default profile) doesn't have authenticated/anon/service_role — skip silently.
        # Fully-static SQL (single triple-quoted literal, no Python concatenation, no
        # PL/pgSQL EXECUTE) so SQL-injection heuristics don't flag this DDL bootstrap.
        # The role names are a hardcoded whitelist.
        conn.execute(sa.text("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'postgres') THEN
                    GRANT USAGE ON SCHEMA extensions TO postgres;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                    GRANT USAGE ON SCHEMA extensions TO authenticated;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                    GRANT USAGE ON SCHEMA extensions TO anon;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
                    GRANT USAGE ON SCHEMA extensions TO service_role;
                END IF;
            END $$
        """))
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions"))
    except Exception:
        # Extension not available (managed Postgres, insufficient privileges, etc.)
        # Roll back the failed statement and skip index creation.
        conn.execute(sa.text("ROLLBACK"))
        conn.execute(sa.text("BEGIN"))
        return

    schema = _get_schema_prefix()
    # GIN index on canonical_name enables sub-millisecond trigram similarity queries
    # (% operator, similarity()) instead of full-table scans across all bank entities.
    op.execute("COMMIT")
    # Ensure `extensions` is in search_path so `gin_trgm_ops` operator class is visible
    # to the CREATE INDEX statement. pg_trgm lives in `extensions` per ADR-019, while
    # alembic's env.py sets search_path to '"<tenant>", public' — without extensions
    # the GIN operator class lookup fails with "gin_trgm_ops does not exist".
    # SET (not SET LOCAL) is session-scoped and persists across CONCURRENTLY's
    # implicit COMMIT/BEGIN dance.
    target_schema = context.config.get_main_option("target_schema")
    if target_schema:
        op.execute(f'SET search_path TO "{target_schema}", public, extensions')
    else:
        op.execute("SET search_path TO public, extensions")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_canonical_name_trgm_idx "
        f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    # nosemgrep: python.sqlalchemy.security.audit.formatted-sql-query.formatted-sql-query
    # `schema` comes from alembic's target_schema config (trusted, set by env.py), not user input.
    # DDL with identifier interpolation cannot use parameterized queries.
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}entities_canonical_name_trgm_idx")
    # Note: not dropping pg_trgm extension as other indexes may depend on it
