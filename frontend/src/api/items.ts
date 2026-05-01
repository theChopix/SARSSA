/**
 * API service for item enrichment and artifact endpoints.
 *
 * This module provides functions to fetch enriched item metadata
 * and raw step artifacts from the backend.
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

/**
 * Build the URL for a raw (non-JSON) artifact served by the backend.
 *
 * The returned URL can be used directly as the `src` of an `<img>`
 * or `<iframe>` element.  The backend serves the file with the
 * correct `Content-Type` header inferred from the filename extension.
 *
 * @param runId    - MLflow run ID of the plugin step.
 * @param filename - Artifact filename (e.g. "dendrogram.svg").
 * @returns Absolute URL string pointing to the raw artifact endpoint.
 */
export function buildRawArtifactUrl(
  runId: string,
  filename: string
): string {
  const params = new URLSearchParams({ run_id: runId, filename });
  return `${API_BASE_URL}/items/artifact-raw?${params}`;
}

/**
 * Fetch a JSON artifact from a plugin step's MLflow run.
 *
 * Calls `GET /items/artifact` which acts as a proxy to MLflow,
 * so the frontend never needs direct MLflow access.
 *
 * @param runId    - MLflow run ID of the plugin step.
 * @param filename - Artifact filename (e.g. "steered_recommendations.json").
 * @returns The parsed JSON content of the artifact.
 *
 * @example
 * ```ts
 * const itemIds = await fetchStepArtifact("abc123", "steered_recommendations.json");
 * // itemIds → ["42", "107", "253"]
 * ```
 *
 * @throws {Error} If the HTTP request fails (non-2xx status).
 */
export async function fetchStepArtifact(
  runId: string,
  filename: string
): Promise<unknown> {
  const params = new URLSearchParams({
    run_id: runId,
    filename,
  });

  const response = await fetch(`${API_BASE_URL}/items/artifact?${params}`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch artifact '${filename}': ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}
