/**
 * Header — shared top navigation bar.
 *
 * Renders the app logo, an in-app Guide link, a GitHub repository
 * link, and an MLflow deep link. Used as the persistent header
 * across all pages via the Layout component.
 *
 * ┌────────────────────────────────────────────────────────────────┐
 * │  SARSSAe     ⓘ guide | ⌥ github |   mlflow … Results (link)   │
 * └────────────────────────────────────────────────────────────────┘
 */

import { Link } from "react-router-dom";
import { Info } from "lucide-react";

import { usePipelineStore, mlflowExperimentUrl } from "../store/pipelineStore";
import { RunningTasksMenu } from "./RunningTasksMenu";
import { ExperimentMenu } from "./ExperimentMenu";

const REPO_URL = "https://github.com/theChopix/SARSSA";

export function Header() {
  const mlflowInfo = usePipelineStore((s) => s.mlflowInfo);

  return (
    <header className="border-b border-gray-200 bg-white px-8 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1>
          <Link
            to="/"
            className="text-lg font-bold tracking-wide text-gray-900 hover:text-gray-700"
          >
            SARSSAe
          </Link>
        </h1>
        <RunningTasksMenu />
      </div>
      <nav className="flex items-center gap-6">
        <Link
          to="/guide"
          className="flex items-center gap-1.5 text-lg font-medium tracking-wide text-blue-500 hover:text-blue-700"
        >
          <Info className="h-5 w-5" />
          guide
        </Link>
        <span
          className="h-5 w-px bg-gray-300"
          aria-hidden="true"
        />
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-lg font-medium tracking-wide text-blue-500 hover:text-blue-700"
        >
          <svg
            className="h-5 w-5"
            viewBox="0 0 16 16"
            fill="currentColor"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
          </svg>
          github
        </a>
        <span
          className="h-5 w-px bg-gray-300"
          aria-hidden="true"
        />
        <ExperimentMenu />
        <span
          className="h-5 w-px bg-gray-300"
          aria-hidden="true"
        />
        {mlflowInfo ? (
          <a
            href={mlflowExperimentUrl(mlflowInfo)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-lg font-medium tracking-wide text-blue-500 hover:text-blue-700"
          >
            <img
              src="/mlflow-logo.png"
              alt="mlflow"
              className="h-5"
            />
            pipeline experiments results
          </a>
        ) : (
          <span
            className="flex items-center gap-1.5 text-lg font-medium tracking-wide text-gray-400 cursor-not-allowed"
            aria-disabled="true"
          >
            <img
              src="/mlflow-logo.png"
              alt="mlflow"
              className="h-5 opacity-50"
            />
            pipeline experiments results
          </span>
        )}
      </nav>
    </header>
  );
}
