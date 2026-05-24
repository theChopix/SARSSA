/**
 * InfoTooltip — an accessible ⓘ icon with a hover/focus tooltip.
 *
 * The trigger is a real <button> so the tooltip is reachable by
 * keyboard (Tab); the bubble reveals on hover *and* on focus-within
 * and is wired to the button via `aria-describedby` for screen
 * readers.
 *
 *   ┌──────────────────────────────┐
 *   │ Tooltip text here            │  ← bubble (above the icon)
 *   └──────────────────────────────┘
 *          ⓘ                          ← focusable trigger
 *
 * Callers must supply an explicit `ariaLabel` describing what the
 * tooltip is about (e.g. "Parameter description",
 * "Category description") so screen-reader announcements are
 * meaningful.
 */

import { useId } from "react";
import { Info } from "lucide-react";

export default function InfoTooltip({
  text,
  ariaLabel,
}: {
  text: string;
  ariaLabel: string;
}) {
  const tooltipId = useId();

  return (
    <span className="relative inline-flex group">
      <button
        type="button"
        aria-label={ariaLabel}
        aria-describedby={tooltipId}
        className="inline-flex items-center justify-center rounded-full
                   text-gray-400 hover:text-gray-600 focus:text-gray-600
                   focus:outline-none focus:ring-2 focus:ring-blue-400
                   cursor-help"
      >
        <Info className="h-3.5 w-3.5" />
      </button>

      <span
        id={tooltipId}
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 mb-1 z-50
                   w-max max-w-xs whitespace-normal break-words rounded
                   bg-gray-800 px-2 py-1 text-xs text-white shadow-lg
                   opacity-0 transition-opacity duration-100
                   group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}
