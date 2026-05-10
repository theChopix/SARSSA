/**
 * Layout — shared page shell rendered around every route.
 *
 * Provides the persistent header and a full-height content area.
 * Child routes are rendered via React Router's `<Outlet />`.
 *
 * ┌──────────────────────────────────────────────────┐
 * │  <Header />                                       │
 * ├──────────────────────────────────────────────────┤
 * │  <Outlet />  ← current route's page component    │
 * └──────────────────────────────────────────────────┘
 */

import { useEffect } from "react";
import { Outlet } from "react-router-dom";
import { Toaster } from "sonner";

import { Header } from "./Header";
import LaunchModal from "./LaunchModal";
import { usePipelineStore } from "../store/pipelineStore";

export function Layout() {
  const loadMlflowInfo = usePipelineStore((s) => s.loadMlflowInfo);

  // The header's MLflow deep link reads mlflowInfo from the store;
  // load it here so the link works on every route (the results page
  // opens in a new tab where the store starts empty).
  useEffect(() => {
    loadMlflowInfo();
  }, [loadMlflowInfo]);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Toaster richColors position="top-center" duration={6000} />
      <LaunchModal />
      <Header />
      <Outlet />
    </div>
  );
}
