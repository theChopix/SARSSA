/**
 * The last thing a running task reported.
 *
 * Two variants, because the two hosts have opposite backgrounds:
 *
 * - "light" (running-tasks menu, white)
 * - "onColor" (blue bottom bar)
 */

import type { TaskMessage } from "../types/pipeline";
import { formatAgo, formatClock } from "../utils/duration";

/** Chip fill + text for the light (white-background) variant. */
const LIGHT_STYLES: Record<string, string> = {
  progress: "bg-slate-100 text-slate-600",
  info: "bg-blue-50 text-blue-700",
  success: "bg-green-50 text-green-700",
  warning: "bg-amber-50 text-amber-800",
  error: "bg-red-50 text-red-700",
};

/** Status-dot colour for the onColor (blue-background) variant. */
const DOT_STYLES: Record<string, string> = {
  progress: "bg-blue-200",
  info: "bg-blue-200",
  success: "bg-green-300",
  warning: "bg-amber-300",
  error: "bg-red-300",
};

export function TaskActivity({
  message,
  variant = "light",
}: {
  message: TaskMessage;
  variant?: "light" | "onColor";
}) {
  const title = `${message.text} (${formatClock(message.timestamp)})`;

  if (variant === "onColor") {
    const dot = DOT_STYLES[message.level] ?? DOT_STYLES.info;
    return (
      <span
        className="inline-flex min-w-0 max-w-full items-center gap-2 text-sm text-white"
        title={title}
      >
        <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} />
        <span className="truncate font-medium">{message.text}</span>
        <span className="shrink-0 text-xs font-normal text-blue-100/80">
          {formatAgo(message.timestamp)}
        </span>
      </span>
    );
  }

  const tint = LIGHT_STYLES[message.level] ?? LIGHT_STYLES.info;
  return (
    <span
      className={`inline-flex min-w-0 max-w-full items-baseline gap-2 rounded
                  px-2 py-0.5 text-xs font-normal ${tint}`}
      title={title}
    >
      <span className="truncate">{message.text}</span>
      <span className="shrink-0 opacity-60">{formatAgo(message.timestamp)}</span>
    </span>
  );
}
