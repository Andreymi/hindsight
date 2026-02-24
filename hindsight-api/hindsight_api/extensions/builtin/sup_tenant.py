"""
SUP (Industry OS) Tenant Extension for Hindsight

Schema-per-tenant isolation for multi-tenant SUP deployments.
Each SUP tenant gets its own PostgreSQL schema (hs_{tenant_id}),
ensuring complete data isolation between organizations.

Authentication uses a compound API key format: {tenant_id}:{server_secret}
passed as a Bearer token. This avoids the need for custom headers or
upstream changes to RequestContext.

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.sup_tenant:SupTenantExtension
    HINDSIGHT_API_TENANT_SERVER_SECRET=your-server-to-server-secret

Usage:
    Clients pass compound key in the Authorization header:

    curl -H "Authorization: Bearer {tenant_id}:{secret}" \\
        http://localhost:8890/v1/default/banks/contact-123/memories/recall

See ADR-004-v2 §13 for architectural context.
"""

from __future__ import annotations

import logging
import re

from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = ["SupTenantExtension"]

# Tenant IDs are UUIDs — validate before using in schema names
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


class SupTenantExtension(TenantExtension):
    """Schema-per-tenant isolation for SUP (Industry OS).

    Auth format: Authorization: Bearer {tenant_id}:{server_secret}
    Each tenant gets schema hs_{tenant_id} with lazy Alembic migrations.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self._server_secret = config.get("server_secret")
        if not self._server_secret:
            raise ValueError(
                "HINDSIGHT_API_TENANT_SERVER_SECRET is required. "
                "Set it to a shared secret between SUP and Hindsight API."
            )
        self._initialized_schemas: set[str] = set()

    async def on_startup(self) -> None:
        logger.info("SUP tenant extension initialized (schema-per-tenant)")

    async def authenticate(self, context: RequestContext) -> TenantContext:
        token = context.api_key or ""
        if ":" not in token:
            raise AuthenticationError("Expected compound key format: {tenant_id}:{secret}")

        tenant_id, secret = token.split(":", 1)

        if secret != self._server_secret:
            raise AuthenticationError("Invalid server secret")

        if not tenant_id:
            raise AuthenticationError("Empty tenant_id in compound key")

        if not _UUID_RE.match(tenant_id):
            raise AuthenticationError("Invalid tenant_id format (expected UUID)")

        # Schema name: hs_{uuid_with_underscores}
        safe_tenant_id = tenant_id.replace("-", "_")
        schema_name = f"hs_{safe_tenant_id}"

        # Lazy migration on first access
        if schema_name not in self._initialized_schemas:
            logger.info("Initializing schema: %s", schema_name)
            try:
                await self.context.run_migration(schema_name)
                self._initialized_schemas.add(schema_name)
                logger.info("Schema ready: %s", schema_name)
            except Exception as e:
                logger.error("Schema initialization failed for %s: %s", schema_name, e)
                raise AuthenticationError(f"Failed to initialize tenant: {e!s}")

        return TenantContext(schema_name=schema_name)

    async def list_tenants(self) -> list[Tenant]:
        return [Tenant(schema=s) for s in self._initialized_schemas]
