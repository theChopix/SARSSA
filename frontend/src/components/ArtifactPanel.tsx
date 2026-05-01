/**
 * ArtifactPanel — renders standalone visual artifacts from a plugin step.
 *
 * Each file in the display spec is rendered according to its
 * `content_type`:
 *
 * - `image/*`    → `<img>` element
 * - `text/html`  → `<iframe>` element (e.g. interactive Plotly charts)
 *
 * The artifact bytes are fetched directly by the browser via the
 * `GET /items/artifact-raw` endpoint (no client-side fetch needed).
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  ── Dendrogram ──────────────────────────────────────── │
 * │  [<img src="...artifact-raw?...dendrogram.svg">]        │
 * │                                                         │
 * │  ── Embedding Map ───────────────────────────────────── │
 * │  [<iframe src="...artifact-raw?...embedding_map.html">] │
 * └─────────────────────────────────────────────────────────┘
 */

import { AlertCircle } from "lucide-react";

import { buildRawArtifactUrl } from "../api/items";
import type { ArtifactFileSpec } from "../types/plugin";

// ── Props ────────────────────────────────────────────────

interface ArtifactPanelProps {
  files: ArtifactFileSpec[];
  stepRunId: string;
}

// ── Single artifact renderer ─────────────────────────────

function ArtifactRenderer({
  file,
  stepRunId,
}: {
  file: ArtifactFileSpec;
  stepRunId: string;
}) {
  const url = buildRawArtifactUrl(stepRunId, file.filename);

  if (file.content_type.startsWith("image/")) {
    return (
      <img
        src={url}
        alt={file.label}
        className="max-w-full rounded border border-gray-200"
      />
    );
  }

  if (file.content_type === "text/html") {
    return (
      <iframe
        src={url}
        title={file.label}
        className="w-full rounded border border-gray-200"
        style={{ height: "600px" }}
        sandbox="allow-scripts allow-same-origin"
      />
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm text-amber-600">
      <AlertCircle className="h-4 w-4" />
      <span>
        Unsupported content type: <code>{file.content_type}</code>
      </span>
    </div>
  );
}

// ── Main component ───────────────────────────────────────

export function ArtifactPanel({ files, stepRunId }: ArtifactPanelProps) {
  if (files.length === 0) {
    return (
      <p className="mt-4 text-sm text-gray-500">
        No artifact files to display.
      </p>
    );
  }

  return (
    <div className="mt-6 space-y-6">
      {files.map((file) => (
        <section key={file.filename}>
          <h3 className="mb-2 text-sm font-semibold text-gray-700">
            {file.label}
          </h3>
          <ArtifactRenderer file={file} stepRunId={stepRunId} />
        </section>
      ))}
    </div>
  );
}
