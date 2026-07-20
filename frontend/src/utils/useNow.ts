import { useEffect, useState } from "react";

import { nowSeconds } from "./duration";

/**
 * Re-render on a timer so relative times ("3m ago") stay current.
 *
 * Store polls only re-render a component when the *selected* value
 * changes, and a task's `started_at` never does — without this tick an
 * elapsed-time label would freeze at whatever it showed on the first
 * render.
 *
 * @param intervalMs - Tick period; 5s is plenty for minute-resolution labels.
 * @returns The current time in epoch seconds.
 */
export function useNow(intervalMs = 5000): number {
  const [now, setNow] = useState(nowSeconds);
  useEffect(() => {
    const id = setInterval(() => setNow(nowSeconds()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
