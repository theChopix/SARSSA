# 🖥️ SARSSA Frontend (`frontend/`)

> **What this is:** a high-level map of the web UI — the React app
> that lets a user configure and run pipelines and browse results. It
> is intentionally an *orientation* doc, not an exhaustive
> component reference.
>
> **Who should read this:** someone who needs to **fix or extend the
> UI**. Note this is the *uncommon* path — SARSSA's normal extension
> point is the backend (a new plugin needs **zero** frontend changes,
> see §5). Read this only if you're touching the interface itself. The
> backend it talks to is documented in
> [`../src/app/README.md`](../src/app/README.md).

---

## 📑 Table of Contents

1. [🧰 Tech stack](#-1-tech-stack)
2. [🚀 Run & build](#-2-run--build)
3. [🗂️ Folder & file map](#-3-folder--file-map)
4. [🔄 How it talks to the backend](#-4-how-it-talks-to-the-backend)
5. [🧭 "I want to change X — where?"](#-5-i-want-to-change-x--where)
6. [⚠️ Gotchas](#-6-gotchas)
7. [➡️ Where to go next](#-7-where-to-go-next)

---

## 🧰 1. Tech stack

| Concern | Choice |
|---------|--------|
| UI library | **React 19** + react-dom |
| Language | **TypeScript** (project-refs: `tsconfig.app.json` / `tsconfig.node.json`) |
| Build / dev server | **Vite 8** (`@vitejs/plugin-react`, HMR) |
| Styling | **Tailwind CSS 4** via `@tailwindcss/vite` — activated by the single `@import "tailwindcss"` in `src/index.css` |
| Routing | **react-router-dom 7** |
| Global state | **Zustand 5** (one store) |
| Toasts | **sonner** |
| Icons | **lucide-react** |
| Lint | **ESLint 9** flat config (`eslint.config.js`: js + typescript-eslint + react-hooks + react-refresh; ignores `dist`) |

npm scripts (`package.json`): `dev` (Vite dev server), `build`
(`tsc -b && vite build`), `lint` (`eslint .`), `preview`.

There is **no Redux, no data-fetching library, no CSS framework beyond
Tailwind** — state is one Zustand store and HTTP is hand-rolled `fetch`
wrappers. Keep it that way unless there's a strong reason.

---

## 🚀 2. Run & build

**Local dev:**
```bash
just frontend-install      # npm install
just frontend-dev          # npm run dev  → http://localhost:5173
```
The dev server runs on **:5173** with hot-module reload. It expects
the backend on **:8000** (see the contract below).

**Production / Docker:** the multi-stage `Dockerfile` builds the app
(`npm run build` → static bundle in `dist/`) and serves it with
**nginx** (`nginx.conf`): SPA fallback (`try_files … /index.html` so
client-side routes work on refresh), long-cache for hashed
`/assets/`, no-cache for the HTML shell. `docker-compose` publishes
the container on host port 5173 and the backend on 8000.

> **⚠️ The 5173 ↔ 8000 / CORS contract — read this.**
> `src/constants.ts` **hardcodes** `API_BASE_URL =
> "http://localhost:8000"`, and the backend's CORS allow-list is
> **only** `http://localhost:5173`. There is **no Vite proxy** — the
> browser calls the backend cross-origin directly. So the UI must be
> served from `:5173` and the backend from `:8000`, or every request
> is rejected. Changing either side means changing *both* (here and
> the backend's CORS config). This is the single most common
> "nothing works" cause.

---

## 🗂️ 3. Folder & file map

Everything lives under `src/` (plus root config: `Dockerfile`,
`nginx.conf`, `vite.config.ts`, `eslint.config.js`, `tsconfig*.json`,
`index.html`).

| Path | Responsibility |
|------|----------------|
| `main.tsx` | Entry point — mounts `<App/>` into `index.html`'s `#root` |
| `App.tsx` | Router + the **HomePage** (the pipeline-config screen) |
| `constants.ts` | `API_BASE_URL` — the **one** place the backend URL lives |
| `api/plugins.ts` | `GET /plugins/registry` + dynamic dropdown choices |
| `api/pipelines.ts` | runs, context, run-async, task polling, cancel, execute-step(-async), mlflow-info |
| `api/items.ts` | item enrichment + artifact proxy/raw-URL helpers |
| `store/pipelineStore.ts` | **The heart** — the single Zustand store: all state + the run/poll orchestration |
| `store/runPersistence.ts` | Persists an in-progress run to **sessionStorage** so a page refresh restores the running layout (see §6) |
| `types/plugin.ts` · `pipeline.ts` · `items.ts` | TypeScript mirrors of the backend models/response shapes |
| `pages/ResultsPage.tsx` | Standalone results view (opens in a new tab for multi-run results) |
| `components/Layout.tsx` | Page shell (header + routed `<Outlet/>`) |
| `components/Header.tsx` | Top bar + MLflow deep link |
| `components/RunningTasksMenu.tsx` | Header pill listing in-flight runs (polls `GET /pipelines/tasks`); click loads a run |
| `components/PipelineCard.tsx` | **The main workhorse** — one data-driven card per category |
| `components/LaunchModal.tsx` | Pre-run dialog: tags + description → MLflow run tags |
| `components/ArtifactPanel.tsx` | Renders image/HTML artifacts (`<img>`/`<iframe>`) |
| `components/VisualResultsPanel.tsx` + `ItemCard.tsx` | Enriched item-card rows (recommendations etc.) |

The `api/` and `types/` folders are deliberately a thin, typed
boundary mirroring the backend; the *only* genuinely complex files are
`store/pipelineStore.ts` (orchestration) and `PipelineCard.tsx`
(the configurable card UI).

---

## 🔄 4. How it talks to the backend

**The one idea to internalise: the UI is registry-driven.** On load,
the store fetches `GET /plugins/registry`, which fully describes every
category, every plugin, every parameter (and which widget renders it —
text / dropdown / slider / past-runs), and how each result should be
displayed (`DisplaySpec`). The frontend renders *itself* from that
description — it hardcodes no plugin, parameter, or category.

Run flow (the `one_time` pipeline):

```
registry loaded ──▶ a PipelineCard per category
   user configures cards (plugin + params)
        │  "Run up to this step"
        ▼
   LaunchModal (tags + description)
        │  confirmLaunch
        ▼
   store.runPipeline ─▶ POST /pipelines/run-async ─▶ { task_id }
        │                (task may wait as "queued" — compute tasks
        │                 run one at a time on the backend)
        └─▶ poll GET /pipelines/tasks/{task_id} every 2 s
              ├─ track the queued state ("Waiting in queue..." bar)
              ├─ update each card's status (running/done/error)
              ├─ update progress (step i / N)
              ├─ stream the plugin's notifier messages as sonner toasts
              └─ on completed/cancelled/error → stop, refresh past runs
```

**Surviving a refresh.** While a run polls, the store mirrors the
`task_id` + card layout into `sessionStorage` (via
`store/runPersistence.ts`) on every tick. On mount, `App.tsx` calls
`store.resumeRun` *after* the registry loads: if a snapshot exists it
restores the running layout and **re-attaches the same poll loop** — the
background task never stopped, the UI had just lost its handle to it. The
snapshot is cleared on any terminal state.

**Running-tasks menu.** A second, coarser poll (`RunningTasksMenu`,
every 6 s) hits `GET /pipelines/tasks` for *all* queued + running tasks
and shows them as a header pill (queued rows carry a clock icon).
Selecting one calls `store.loadRunningTask`, which
makes it the active run — restoring cards from the session snapshot when
present, else rebuilding them from the task's `steps_requested` (so a run
started in another tab still loads). Only one detailed poller runs at a
time: starting/loading a run supersedes the previous loop.

`multi_run` plugins (inspection, steering, labeling-evaluation) use
the same shape via `store.runSingleStep` → `execute-step-async` → poll,
and their visual output renders inline or in `ResultsPage` (new tab).
"Load from previous run" restores card state from a past run's
`context.json` so you can branch off an existing pipeline.

The frontend **never touches MLflow directly** — the backend proxies
artifacts (`/items/artifact`, `/items/artifact-raw`) and the UI only
builds MLflow *deep links* from `/pipelines/mlflow-info`.

---

## 🧭 5. "I want to change X — where?"

| Goal | Where |
|------|-------|
| Point the UI at a different backend URL | `src/constants.ts` (**and** update the backend CORS allow-list — see §2) |
| Added a backend plugin / parameter / visualization | **Nothing here.** It appears automatically (registry-driven). Only touch the FE for a brand-new *widget type* or *`DisplaySpec` variant* |
| New widget type for a parameter | `types/plugin.ts` (extend `WidgetConfig`) + the widget rendering in `PipelineCard.tsx` |
| New result-display variant | `types/plugin.ts` (`DisplaySpec` union) + a panel component + wire it in `ResultsPage.tsx` |
| New backend call | A typed wrapper in `src/api/` + a type in `src/types/` |
| Global state / run-flow behaviour | `store/pipelineStore.ts` |
| A new page / route | `App.tsx` (routes) + `src/pages/` |
| Styling | Tailwind utility classes inline; `index.css` only for base styles |

The headline: **adding capability is a backend job; the frontend only
changes when the *shape* of the contract changes** (a new widget or
display kind), or for pure UI/UX work.

---

## ⚠️ 6. Gotchas

- **The 5173 ↔ 8000 / CORS contract** (§2) — by far the most common
  source of "the UI loads but every request fails."
- **`ResultsPage` opens in a new browser tab**, which has a *fresh,
  empty* Zustand store. It is written to be self-sufficient (re-reads
  the registry, takes everything else from URL/query params). Don't
  assume any store state carries across tabs.
- **Registry-driven, so don't hardcode.** Resist adding plugin- or
  category-specific branches in components; if you find yourself
  special-casing a plugin name in the UI, the logic almost certainly
  belongs in the backend registry/`io_spec` instead.
- **`types/` must stay in sync with the backend models.** They are
  hand-maintained mirrors of `src/app/models/` — if a backend
  response shape changes, update the matching type or the cast lies.
- **Progress is polling, not websockets** — a fixed 2 s interval in
  the store. Plugin messages surface as toasts only as fast as the
  next poll.
- **Refresh-recovery is sessionStorage-scoped and `one_time`-only.**
  An in-progress *pipeline* run survives a page refresh
  (`store/runPersistence.ts` → `resumeRun`), but mind the edges: it's
  **sessionStorage** (a refresh survives; *closing the tab* does not),
  only the full `runPipeline` flow is persisted (**`runSingleStep` /
  `multi_run` steps are not** recovered across a refresh), and if the
  backend restarted in the meantime the `task_id` 404s → the UI resets to
  idle with a notice. Snapshots are keyed by `task_id` (deliberate
  groundwork for a future "running tasks" list).

---

## ➡️ 7. Where to go next

- ⚙️ **The backend it calls:**
  [`../src/app/README.md`](../src/app/README.md) — the API, the
  pipeline engine, the async task model the polling mirrors.
- 🔌 **The plugin contract behind the registry:**
  [`../src/plugins/README.md`](../src/plugins/README.md) — what
  `io_spec` / parameters / `DisplaySpec` mean (this is what drives
  every widget and result panel here).
- 📘 **Project overview, setup & Docker:**
  [root `README.md`](../README.md).
