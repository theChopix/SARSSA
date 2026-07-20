/**
 * Duration formatting for live task progress.
 *
 * Backend timestamps are Unix epoch *seconds* (Python `time.time()`),
 * not milliseconds — divide `Date.now()` before comparing.
 */

/** Current time in epoch seconds, matching the backend's timestamps. */
export function nowSeconds(): number {
  return Date.now() / 1000;
}

/**
 * Format an elapsed span as a compact "1h 12m" / "45s".
 *
 * Seconds are dropped once the span passes a minute — at that scale
 * they only add noise to a value that refreshes every few seconds.
 *
 * @param seconds - Elapsed time. Negative values clamp to 0.
 */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/**
 * Format a past instant as "just now" / "3m ago".
 *
 * @param timestamp - Epoch seconds.
 */
export function formatAgo(timestamp: number): string {
  const elapsed = nowSeconds() - timestamp;
  if (elapsed < 10) return "just now";
  return `${formatDuration(elapsed)} ago`;
}

/** Format an epoch-seconds instant as a local wall-clock time. */
export function formatClock(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString();
}

/**
 * How long a task, and its current step, have been going.
 *
 * Both are reported because a pipeline's total runtime says little
 * about the step you are actually waiting on. A queued task has
 * neither timestamp yet and reports its wait instead.
 *
 * @param now           - Current epoch seconds (from {@link useNow}).
 * @param createdAt     - When the task was submitted.
 * @param startedAt     - When the worker picked it up; null while queued.
 * @param stepStartedAt - When the current step began; null between steps.
 */
export function taskTimings(
  now: number,
  createdAt: number | null,
  startedAt: number | null,
  stepStartedAt: number | null
): string {
  if (startedAt === null) {
    return createdAt === null ? "" : `queued ${formatDuration(now - createdAt)}`;
  }
  const pipeline = `pipeline ${formatDuration(now - startedAt)}`;
  return stepStartedAt === null
    ? pipeline
    : `${pipeline} · step ${formatDuration(now - stepStartedAt)}`;
}
