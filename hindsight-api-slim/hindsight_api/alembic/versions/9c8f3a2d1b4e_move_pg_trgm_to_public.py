"""Move pg_trgm extension to public schema if it lives elsewhere.

PATCHED migration (not in upstream). Fixes deployments where `pg_trgm` ended up
in a tenant schema instead of `public` — the `%` operator and `similarity()`
functions are then invisible to any session whose search_path doesn't include
that tenant schema, causing background workers (batch_retain, consolidation) to
fail with `operator does not exist: text % text`.

How this happens: SupTenantExtension runs migrations per-tenant with
`SET search_path TO "<tenant_schema>", public`. The upstream c1a2b3d4e5f6
migration's CREATE-EXTENSION statement for pg_trgm omitted an explicit
schema clause, so the extension was created in the first writable schema of
the search_path — the tenant schema. Worker pool connections use the default
postgres role search_path (`"$user", public, extensions`) and never see the
operator.

This migration is idempotent across multiple tenants and a no-op when the
extension is already in `public` or not installed at all.

Revision ID: 9c8f3a2d1b4e
Revises: 8c6fa6f7230b
Create Date: 2026-05-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9c8f3a2d1b4e"
down_revision: str | Sequence[str] | None = "8c6fa6f7230b"
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

    if current_schema == "public":
        return

    op.execute("ALTER EXTENSION pg_trgm SET SCHEMA public")


def downgrade() -> None:
    # No-op: location is design, not data. Restoring the broken placement would
    # re-introduce the worker-visibility bug this migration fixes.
    pass
