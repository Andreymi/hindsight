#!/bin/bash
# Start Hindsight API for SUP (Industry OS) integration.
# Connects to self-hosted Supabase PostgreSQL, uses SupTenantExtension.
#
# Prerequisites:
#   - bunx supabase start (Supabase PG on :54322)
#   - .env with LLM keys (sourced automatically)
#
# Usage:
#   ./scripts/dev/start-api-sup.sh
#   ./scripts/dev/start-api-sup.sh --port 8890

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source LLM keys from .env
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

# SUP-specific config
export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-postgresql://postgres:postgres@localhost:54322/postgres}"
export HINDSIGHT_API_TENANT_EXTENSION="hindsight_api.extensions.builtin.sup_tenant:SupTenantExtension"
export HINDSIGHT_API_TENANT_SERVER_SECRET="${HINDSIGHT_API_TENANT_SERVER_SECRET:-dev-secret}"

# Default port 8890 to avoid conflict with embed daemon on 8888
PORT="${1:-8890}"
if [ "$1" = "--port" ] && [ -n "${2:-}" ]; then
    PORT="$2"
fi

echo "Starting Hindsight API for SUP on port $PORT"
echo "  Database: $HINDSIGHT_API_DATABASE_URL"
echo "  Extension: $HINDSIGHT_API_TENANT_EXTENSION"
echo ""

cd "$ROOT_DIR/hindsight-api"
exec uv run hindsight-api --port "$PORT" "$@"
