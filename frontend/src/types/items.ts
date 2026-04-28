/**
 * TypeScript types for item enrichment.
 *
 * These mirror the response shape of `GET /items/enrich` on the backend.
 * The backend joins raw item IDs with dataset-specific metadata
 * (title, year, genres, image_url, etc.) and returns enriched records.
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  Backend (Python)            →   Frontend (TypeScript)  │
 * │  enrich_items() response     →   EnrichResponse         │
 * │  single enriched item dict   →   EnrichedItem           │
 * └─────────────────────────────────────────────────────────┘
 */

// ── Single enriched item ─────────────────────────────────

/**
 * An item enriched with display-ready metadata.
 *
 * - `id`        – The raw item ID (always present).
 * - `title`     – Human-readable name (always present; falls back to id).
 * - `year`      – Release year (dataset-specific, optional).
 * - `image_url` – Poster / cover image URL (optional).
 * - `genres`    – Genre tags (optional).
 * - `artist`    – Artist name for music datasets (optional).
 *
 * Additional dataset-specific fields may appear via the index signature.
 */
export interface EnrichedItem {
  id: string;
  title: string;
  year?: number;
  image_url?: string;
  genres?: string[];
  artist?: string;
  [key: string]: unknown;
}

// ── Enrichment API response ──────────────────────────────

/**
 * Response shape of `GET /items/enrich`.
 *
 * - `items`              – Enriched item records, one per requested ID
 *                          (order preserved).
 * - `metadata_available` – `true` when the dataset-loading run had an
 *                          `item_metadata.json` artifact; `false` means
 *                          all items fell back to id-as-title.
 */
export interface EnrichResponse {
  items: EnrichedItem[];
  metadata_available: boolean;
}
