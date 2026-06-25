/**
 * Session-scoped persistence for an in-progress pipeline run, so a page
 * refresh can restore the running layout instead of dropping back to the
 * default state while the backend task keeps going.
 *
 * Snapshots are keyed by task_id (a map, not a single slot), so the structure
 * already supports several concurrent runs — groundwork for a future
 * "running tasks" menu — even though the refresh-recovery flow restores only
 * the most recent one. All access goes through this module, so swapping the
 * backing store (sessionStorage ↔ localStorage) or adding backend-driven
 * discovery later stays a localised change.
 */

import type { CardState } from "./pipelineStore";

const STORAGE_KEY = "sarssa.runSnapshots.v1";

/** Everything needed to rebuild the running layout after a refresh. */
export interface RunSnapshot {
  /** Background task id, used to resume polling. */
  taskId: string;
  /** Full per-category card map at snapshot time (config + live status). */
  cards: Record<string, CardState>;
  /** Parent MLflow run id, once known. */
  currentRunId: string | null;
  /** 0-based index of the executing step. */
  currentStepIndex: number;
  /** Total number of steps in the run. */
  totalSteps: number;
  /** Epoch ms when the snapshot was written (used to pick the latest). */
  savedAt: number;
}

function readMap(): Record<string, RunSnapshot> {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object"
      ? (parsed as Record<string, RunSnapshot>)
      : {};
  } catch {
    return {};
  }
}

function writeMap(map: Record<string, RunSnapshot>): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Storage unavailable or full — recovery is best-effort, so ignore.
  }
}

/** Insert or update the snapshot for its task. */
export function saveRunSnapshot(snapshot: RunSnapshot): void {
  const map = readMap();
  map[snapshot.taskId] = snapshot;
  writeMap(map);
}

/** All persisted snapshots, newest first. */
export function loadRunSnapshots(): RunSnapshot[] {
  return Object.values(readMap()).sort((a, b) => b.savedAt - a.savedAt);
}

/** The most recently updated snapshot, or `null` if none. */
export function loadLatestRunSnapshot(): RunSnapshot | null {
  return loadRunSnapshots()[0] ?? null;
}

/** The snapshot for a specific task, or `null` if absent. */
export function loadRunSnapshot(taskId: string): RunSnapshot | null {
  return readMap()[taskId] ?? null;
}

/** Drop the snapshot for one task (no-op if absent). */
export function clearRunSnapshot(taskId: string): void {
  const map = readMap();
  if (taskId in map) {
    delete map[taskId];
    writeMap(map);
  }
}
