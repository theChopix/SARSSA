/**
 * Header — shared top navigation bar.
 *
 * Renders the app logo and an MLflow deep link. Used as the
 * persistent header across all pages via the Layout component.
 *
 * ┌──────────────────────────────────────────────────┐
 * │  SARSSAe               mlflow … Results (link)   │
 * └──────────────────────────────────────────────────┘
 */

import { usePipelineStore, mlflowExperimentUrl } from "../store/pipelineStore";

export function Header() {
  const mlflowInfo = usePipelineStore((s) => s.mlflowInfo);

  return (
    <header className="border-b border-gray-200 bg-white px-8 py-4 flex items-center justify-between">
      <h1 className="text-lg font-bold text-gray-900">SARSSAe</h1>
      {mlflowInfo ? (
        <a
          href={mlflowExperimentUrl(mlflowInfo)}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-blue-500 hover:text-blue-700"
        >
          <img
            src="/mlflow-logo.png"
            alt="mlflow"
            className="h-5"
          />
          Pipeline Experiments Results
        </a>
      ) : (
        <span
          className="flex items-center gap-2 text-sm text-gray-400 cursor-not-allowed"
          aria-disabled="true"
        >
          <img
            src="/mlflow-logo.png"
            alt="mlflow"
            className="h-5 opacity-50"
          />
          Pipeline Experiments Results
        </span>
      )}
    </header>
  );
}
