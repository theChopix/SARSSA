/**
 * API service for plugin-related endpoints.
 *
 * This module contains a single function that fetches the plugin
 * registry from the backend. It is the frontend's equivalent of
 * calling `GET /plugins/registry`.
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  How fetch wrappers work                                 │
 * │                                                          │
 * │  1. We call `fetch(url)` — the browser's built-in HTTP   │
 * │     client. It returns a `Promise<Response>`.             │
 * │                                                          │
 * │  2. A Promise is a value that doesn't exist yet — it     │
 * │     will resolve when the HTTP response arrives.          │
 * │     `await` pauses until it does.                         │
 * │                                                          │
 * │  3. `response.json()` also returns a Promise (parsing     │
 * │     the body is async), so we `await` that too.           │
 * │                                                          │
 * │  4. We cast the result to our TypeScript type so the      │
 * │     rest of the app gets autocomplete & type checking.    │
 * └──────────────────────────────────────────────────────────┘
 */

import { API_BASE_URL } from "../constants";
import type { PluginRegistry } from "../types/plugin";

/**
 * Fetch the full plugin registry from the backend.
 *
 * @returns A record mapping category key → CategoryRegistryEntry.
 *
 * @example
 * ```ts
 * const registry = await fetchPluginRegistry();
 * // registry["dataset_loading"].implementations[0].display_name
 * // → "Lastfm1k Loader"
 * ```
 *
 * @throws {Error} If the HTTP request fails (non-2xx status).
 */
export async function fetchPluginRegistry(): Promise<PluginRegistry> {
  const response = await fetch(`${API_BASE_URL}/plugins/registry`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch plugin registry: ${response.status} ${response.statusText}`
    );
  }

  return (await response.json()) as PluginRegistry;
}
