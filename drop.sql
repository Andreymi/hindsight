-- ============================================================================
-- DROP ALL TABLES FOR MEMORY POC DATABASE
-- ============================================================================
--
-- WARNING: This will completely remove all tables and data!
-- Use with caution, especially in production environments.
--
-- Usage:
--   psql -d your_database -f drop.sql
--
-- After running this, you'll need to recreate the schema:
--   psql -d your_database -f schema.sql
-- ============================================================================

-- Drop all tables in reverse dependency order
-- CASCADE ensures dependent objects are also dropped
DROP TABLE IF EXISTS memory_links CASCADE;
DROP TABLE IF EXISTS entity_cooccurrences CASCADE;
DROP TABLE IF EXISTS unit_entities CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS memory_units CASCADE;

