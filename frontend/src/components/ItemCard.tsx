/**
 * ItemCard — a single enriched item rendered as a compact card.
 *
 * Displays a poster image (or a letter-placeholder when no image is
 * available), the item title, and optional metadata like year and
 * genre tags. The card adapts to whatever fields are present in the
 * `EnrichedItem`.
 *
 * ┌──────────────────────────┐
 * │  ┌────────────────────┐  │
 * │  │                    │  │
 * │  │   poster / letter  │  │
 * │  │                    │  │
 * │  └────────────────────┘  │
 * │  Title (Year)            │
 * │  Artist                  │
 * │  [Genre] [Genre]         │
 * └──────────────────────────┘
 */

import type { EnrichedItem } from "../types/items";

// ── Props ────────────────────────────────────────────────

interface ItemCardProps {
  item: EnrichedItem;
}

// ── Placeholder ──────────────────────────────────────────

/**
 * A coloured circle with the first letter of the title.
 * Shown when no `image_url` is available.
 */
function LetterPlaceholder({ title }: { title: string }) {
  const letter = title.charAt(0).toUpperCase() || "?";
  return (
    <div
      className="w-full aspect-[2/3] rounded-md bg-gray-200
                 flex items-center justify-center"
    >
      <span className="text-3xl font-semibold text-gray-400">{letter}</span>
    </div>
  );
}

// ── Component ────────────────────────────────────────────

export function ItemCard({ item }: ItemCardProps) {
  const { title, year, image_url, genres, artist } = item;

  return (
    <div
      className="flex-shrink-0 w-36 rounded-lg border border-gray-200
                 bg-white shadow-sm overflow-hidden"
    >
      {/* Poster / placeholder */}
      {image_url ? (
        <img
          src={image_url}
          alt={title}
          className="w-full aspect-[2/3] object-cover"
          loading="lazy"
        />
      ) : (
        <LetterPlaceholder title={title} />
      )}

      {/* Text content */}
      <div className="p-2 space-y-0.5">
        {/* Title + year */}
        <p className="text-xs font-medium text-gray-800 leading-tight line-clamp-2">
          {title}
          {year != null && (
            <span className="text-gray-400 font-normal"> ({year})</span>
          )}
        </p>

        {/* Artist (music datasets) */}
        {artist && (
          <p className="text-[11px] text-gray-500 leading-tight truncate">
            {artist}
          </p>
        )}

        {/* Genre tags */}
        {genres && genres.length > 0 && (
          <div className="flex flex-wrap gap-0.5 pt-0.5">
            {genres.slice(0, 3).map((genre) => (
              <span
                key={genre}
                className="px-1.5 py-0.5 text-[10px] rounded-full
                           bg-blue-50 text-blue-600"
              >
                {genre}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
