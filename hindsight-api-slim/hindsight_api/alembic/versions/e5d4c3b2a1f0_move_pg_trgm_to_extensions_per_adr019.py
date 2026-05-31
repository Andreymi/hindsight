"""Move pg_trgm extension to `extensions` schema per sup ADR-019.

PATCHED migration (not in upstream). Continuation of 9c8f3a2d1b4e which moved
pg_trgm to `public` — refactored to `extensions` to align with sup ADR-019
(schema-isolation: domain tables in app, extensions in `extensions`, public
stays empty). Canonical Supabase pattern: pg_cron / pg_net / pgcrypto /
uuid-ossp / pg_stat_statements all live in `extensions` (see
umbrella-sup/sup/supabase/migrations/00000000000000_initial.sql).

Idempotent across reruns:
- If pg_trgm already in `extensions` → no-op.
- If pg_trgm in `public` (installs that stopped at 9c8f3a2d1b4e) → ALTER moves it.
- If pg_trgm in a tenant schema (legacy installs that skipped 9c8f3a2d1b4e) →
  ALTER moves it.
- If pg_trgm not installed (managed PG without trgm) → no-op (upstream #626
  fallback in EntityResolver continues to apply).

Why both 9c8f3a2d1b4e and this migration exist: 9c8f3a2d1b4e was applied in
production deployments before ADR-019 was retrofitted to the patched fork.
Keeping the chain history intact (rather than rewriting 9c8f3a2d1b4e) lets
existing tenant schemas advance to the new head without alembic reorganization.

Revision ID: e5d4c3b2a1f0
Revises: 9c8f3a2d1b4e
Create Date: 2026-06-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5d4c3b2a1f0"
down_revision: str | Sequence[str] | None = "9c8f3a2d1b4e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()

    current_schema = conn.execute(
        sa.text(
            "SELECT n.nspname FROM pg_extension e "
            "JOIN pg_namespace n ON e.extnamespace = n.oid "
            "WHERE e.extname = 'pg_trgm'"
        )
    ).scalar()

    if current_schema is None:
        return

    if current_schema == "extensions":
        return

    conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS extensions"))
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

    op.execute("ALTER EXTENSION pg_trgm SET SCHEMA extensions")


def downgrade() -> None:
    pass
