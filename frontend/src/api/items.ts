/**
 * API service for item enrichment endpoints.
 *
 * This module provides functions to fetch enriched item metadata
 * from the backend. It is the frontend's equivalent of calling
 * `GET /items/enrich`.
 */

import { API_BASE_URL } from "../constants";
import type { EnrichResponse } from "../types/items";

/**
 * Fetch enriched metadata for a list of item IDs.
 *
 * Calls `GET /items/enrich` with the dataset-loading run ID and
 * a comma-separated list of item IDs. The backend joins the IDs
 * with `item_metadata.json` from the given MLflow run.
 *
 * @param runId  - MLflow run ID of the dataset-loading step.
 * @param ids    - Array of item IDs to enrich.
 * @returns Enriched items and a flag indicating metadata availability.
 *
 * @example
 * ```ts
 * const { items, metadata_available } = await fetchEnrichedItems(
 *   "abc123",
 *   ["42", "107", "253"]
 * );
 * ```
 *
 * @throws {Error} If the HTTP request fails (non-2xx status).
 */
export async function fetchEnrichedItems(
  runId: string,
  ids: string[]
): Promise<EnrichResponse> {
  const params = new URLSearchParams({
    run_id: runId,
    ids: ids.join(","),
  });

  const response = await fetch(`${API_BASE_URL}/items/enrich?${params}`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch enriched items: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as EnrichResponse;
}
