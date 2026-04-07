/**
 * Root application component.
 *
 * React components are **functions that return JSX** (HTML-like syntax).
 * This file is the top-level component rendered by `main.tsx`.
 *
 * The `className` strings (e.g. `"bg-gray-950"`) are **TailwindCSS
 * utility classes**. Each one maps to a single CSS property:
 *
 *   bg-gray-950   → background-color: #030712
 *   min-h-screen  → min-height: 100vh
 *   text-white    → color: #fff
 *   p-8           → padding: 2rem  (8 × 0.25rem)
 *   text-3xl      → font-size: 1.875rem
 *   font-bold     → font-weight: 700
 *   text-gray-400 → color: #9ca3af
 *
 * You chain them together to style elements without writing CSS files.
 */

/**
 * App — the top-level React component.
 *
 * Right now it renders a placeholder header. In later phases we will
 * add the pipeline card grid, SSE streaming status, and run history.
 */
function App() {
  return (
    <div className="bg-gray-950 min-h-screen text-white">
      {/* ── Header ─────────────────────────────────── */}
      <header className="border-b border-gray-800 px-8 py-6">
        <h1 className="text-2xl font-bold tracking-tight">
          SARSSA
        </h1>
        <p className="text-sm text-gray-400 mt-1">
          SAE-based Recommender System Steering &amp; Analysis
        </p>
      </header>

      {/* ── Main content area (placeholder) ────────── */}
      <main className="p-8">
        <p className="text-gray-500">
          Pipeline cards will appear here.
        </p>
      </main>
    </div>
  );
}

export default App;
