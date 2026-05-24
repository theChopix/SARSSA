/**
 * GuidePage — in-app user guide.
 *
 * Standalone route at `/guide`, reached via the "Guide" link in the
 * Header. Self-contained: no store reads, no async loads — the page
 * is pure static JSX so it works in any tab (including the new tab
 * the results page opens into) without needing the registry loaded.
 */

export function GuidePage() {
  return (
    <div className="flex-1 px-8 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Guide</h1>
    </div>
  );
}
