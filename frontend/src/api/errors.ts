/**
 * Typed error thrown by the API client when a fetch responds with a
 * non-2xx status. Callers use `instanceof ApiError` and inspect
 * `.status` to branch on HTTP status (e.g. 404 vs 500) without parsing
 * the message string.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly detail?: string;

  constructor(
    status: number,
    statusText: string,
    context: string,
    detail?: string
  ) {
    const tail = detail ? ` — ${detail}` : "";
    super(`${context}: ${status} ${statusText}${tail}`);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.detail = detail;
  }

  /**
   * Build an ApiError from a non-OK Response, attaching the server's
   * `detail` field (FastAPI convention) when the body is JSON.
   */
  static async fromResponse(
    response: Response,
    context: string
  ): Promise<ApiError> {
    let detail: string | undefined;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // body wasn't JSON; carry on without detail
    }
    return new ApiError(
      response.status,
      response.statusText,
      context,
      detail
    );
  }
}
