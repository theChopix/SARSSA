/**
 * Application entry point.
 *
 * This file is the first TypeScript that runs in the browser.
 * It does three things:
 *
 * 1. Imports `index.css` — which activates TailwindCSS.
 * 2. Finds the `<div id="root">` element in `index.html`.
 * 3. Renders the `<App />` component into that element.
 *
 * `StrictMode` is a React wrapper that enables extra development
 * warnings (double-renders, deprecated API usage, etc.).
 * It has no effect in production builds.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
