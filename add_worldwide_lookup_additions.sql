-- Additive lookup seed for worldwide garment support.
-- Safe behavior:
-- 1. Inserts only missing keys.
-- 2. Does not update or delete existing rows.
-- 3. Resolves parent_id from existing parent keys already present in lookups.
--
-- Assumptions:
-- - PostgreSQL
-- - lookups(id, type, key, label, parent_id, is_active, created_at, updated_at)
-- - gen_random_uuid() is available; if not, replace with uuid_generate_v4()

BEGIN;

WITH seed(parent_key, key, label) AS (
    VALUES
        ('tops', 'bras', 'Bras'),
        ('tops', 'bralettes', 'Bralettes'),
        ('tops', 'bustiers', 'Bustiers'),
        ('tops', 'corsets', 'Corsets'),
        ('tops', 'tunics', 'Tunics'),
        ('tops', 'kurtas', 'Kurtas'),
        ('tops', 'kurtis', 'Kurtis'),

        ('bottoms', 'palazzos', 'Palazzos'),
        ('bottoms', 'salwars', 'Salwars'),
        ('bottoms', 'dhoti_pants', 'Dhoti Pants'),

        ('skirts', 'lehenga_skirts', 'Lehenga Skirts'),

        ('dresses', 'gowns', 'Gowns'),
        ('dresses', 'slip_dresses', 'Slip Dresses'),
        ('dresses', 'bodycon_dresses', 'Bodycon Dresses'),
        ('dresses', 'rompers', 'Rompers'),
        ('dresses', 'kaftans', 'Kaftans'),
        ('dresses', 'anarkalis', 'Anarkalis'),
        ('dresses', 'sarees', 'Sarees'),
        ('dresses', 'lehenga_sets', 'Lehenga Sets'),
        ('dresses', 'salwar_kameez_sets', 'Salwar Kameez Sets'),
        ('dresses', 'abayas', 'Abayas'),
        ('dresses', 'qipaos', 'Qipaos'),

        ('outerwear', 'boleros', 'Boleros'),
        ('outerwear', 'shrugs', 'Shrugs'),
        ('outerwear', 'capes', 'Capes'),
        ('outerwear', 'ponchos', 'Ponchos'),
        ('outerwear', 'shawls', 'Shawls'),
        ('outerwear', 'dupattas', 'Dupattas'),
        ('outerwear', 'kimonos', 'Kimonos')
),
resolved AS (
    SELECT
        gen_random_uuid() AS id,
        'CATEGORY'::text AS type,
        s.key,
        s.label,
        p.id AS parent_id,
        TRUE AS is_active,
        NOW() AS created_at,
        NULL::timestamp AS updated_at
    FROM seed s
    JOIN lookups p
      ON p.key = s.parent_key
     AND p.type = 'CATEGORY'
)
INSERT INTO lookups (
    id,
    type,
    key,
    label,
    parent_id,
    is_active,
    created_at,
    updated_at
)
SELECT
    r.id,
    r.type,
    r.key,
    r.label,
    r.parent_id,
    r.is_active,
    r.created_at,
    r.updated_at
FROM resolved r
WHERE NOT EXISTS (
    SELECT 1
    FROM lookups existing
    WHERE existing.key = r.key
      AND existing.type = 'CATEGORY'
);

COMMIT;
