/**
 * Application-wide constants.
 *
 * Centralised here so that if the backend URL changes (e.g. in
 * production), you only need to update one place.
 */

/**
 * Base URL of the FastAPI backend.
 *
 * During development the backend runs on port 8000 and the Vite
 * dev server on port 5173. The CORS middleware on the backend
 * is configured to accept requests from `http://localhost:5173`.
 */
export const API_BASE_URL = "http://localhost:8000";
