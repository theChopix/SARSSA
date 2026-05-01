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

import { Outlet } from "react-router-dom";
import { Toaster } from "sonner";

import { Header } from "./Header";
import LaunchModal from "./LaunchModal";

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Toaster richColors position="top-center" duration={6000} />
      <LaunchModal />
      <Header />
      <Outlet />
    </div>
  );
}
