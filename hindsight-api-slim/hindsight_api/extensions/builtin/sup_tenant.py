"""
SUP (Industry OS) Tenant Extension for Hindsight

Schema-per-org isolation for multi-tenant SUP deployments.
Each SUP tenant (organization) gets its own PostgreSQL schema (hs_{tenant_id}),
ensuring complete data isolation between organizations.

Authentication uses Supabase JWTs verified locally via JWKS public keys
(no network call per request). The target tenant is specified via the
X-Tenant-Id header. The extension verifies that the authenticated user
has access to the requested tenant by checking the JWT's
tenant_ids claim (set by Supabase custom_access_token_hook).

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.sup_tenant:SupTenantExtension
    HINDSIGHT_API_TENANT_SUPABASE_URL=http://localhost:54321

Usage:
    curl -H "Authorization: Bearer <supabase_jwt>" \\
         -H "X-Tenant-Id: <tenant_uuid>" \\
        http://localhost:8890/v1/default/banks/my-bank/memories/recall
"""

from __future__ import annotations

import logging
import re
import time

import httpx
import jwt as pyjwt
from jwt import PyJWK

from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = ["SupTenantExtension"]

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

MIN_TOKEN_LENGTH = 20
REQUEST_TIMEOUT_SECONDS = 10.0
JWKS_CACHE_TTL_SECONDS = 600
JWKS_MIN_REFRESH_INTERVAL_SECONDS = 30
SUPPORTED_ALGORITHMS = ["RS256", "ES256"]


class SupTenantExtension(TenantExtension):
    """Schema-per-org isolation for SUP (Industry OS).

    Auth: Supabase JWT (JWKS-verified) + X-Tenant-Id header.
    Each tenant gets schema hs_{tenant_id} with lazy Alembic migrations.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)

        self.supabase_url = (config.get("supabase_url") or "").rstrip("/")
        if not self.supabase_url:
            raise ValueError(
                "HINDSIGHT_API_TENANT_SUPABASE_URL is required. "
                "Set it to your Supabase project URL (e.g., http://localhost:54321)"
            )

        self._initialized_schemas: set[str] = set()
        self._http_client: httpx.AsyncClient | None = None
        self._jwks_keys: dict[str, PyJWK] = {}
        self._jwks_last_fetched: float = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        logger.info("SUP tenant extension initializing (schema-per-org)")
        logger.info("Supabase URL: %s", self.supabase_url)
        self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        await self._fetch_jwks()
        if not self._jwks_keys:
            raise ValueError(
                "JWKS endpoint returned no signing keys. "
                "Ensure Supabase Auth uses asymmetric JWT signing (RS256/ES256)."
            )
        logger.info("JWKS loaded — %d key(s)", len(self._jwks_keys))

    async def on_shutdown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # JWKS
    # ------------------------------------------------------------------

    async def _fetch_jwks(self) -> None:
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        url = f"{self.supabase_url}/auth/v1/.well-known/jwks.json"
        response = await self._http_client.get(url)
        response.raise_for_status()

        jwks_data = response.json()
        keys: dict[str, PyJWK] = {}
        for key_data in jwks_data.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                keys[kid] = PyJWK(key_data)

        self._jwks_keys = keys
        self._jwks_last_fetched = time.monotonic()

    async def _get_signing_key(self, token: str) -> PyJWK:
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise AuthenticationError("Token missing key ID (kid) header")

        now = time.monotonic()
        if now - self._jwks_last_fetched > JWKS_CACHE_TTL_SECONDS:
            await self._fetch_jwks()

        if kid in self._jwks_keys:
            return self._jwks_keys[kid]

        # Key rotation: one forced refresh
        if now - self._jwks_last_fetched > JWKS_MIN_REFRESH_INTERVAL_SECONDS:
            logger.info("Signing key %s not in cache, refreshing JWKS", kid)
            await self._fetch_jwks()
            if kid in self._jwks_keys:
                return self._jwks_keys[kid]

        raise AuthenticationError("Unable to find signing key for token")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self, context: RequestContext) -> TenantContext:
        token = context.api_key or ""
        if not token or len(token) < MIN_TOKEN_LENGTH:
            raise AuthenticationError("Missing or invalid Authorization header")

        # 1. Verify JWT
        try:
            signing_key = await self._get_signing_key(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=SUPPORTED_ALGORITHMS,
                audience="authenticated",
                issuer=f"{self.supabase_url}/auth/v1",
            )
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except pyjwt.InvalidAudienceError:
            raise AuthenticationError("Invalid token audience")
        except pyjwt.InvalidIssuerError:
            raise AuthenticationError("Invalid token issuer")
        except pyjwt.DecodeError:
            raise AuthenticationError("Invalid token")
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {e!s}")

        # 2. Extract tenant from X-Tenant-Id header
        headers = context.headers or {}
        tenant_id = headers.get("x-tenant-id", "").strip()
        if not tenant_id:
            raise AuthenticationError("Missing X-Tenant-Id header")

        if not _UUID_RE.match(tenant_id):
            raise AuthenticationError("Invalid X-Tenant-Id format (expected UUID)")

        # 3. Authorize: user must have access to this tenant
        # Supabase custom_access_token_hook places tenant_ids in top-level claims
        allowed_tenants = payload.get("tenant_ids", [])
        if not isinstance(allowed_tenants, list) or tenant_id not in allowed_tenants:
            raise AuthenticationError(f"User does not have access to tenant {tenant_id}")

        # 4. Resolve schema
        safe_tenant_id = tenant_id.replace("-", "_")
        schema_name = f"hs_{safe_tenant_id}"

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
