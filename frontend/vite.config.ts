/**
 * Vite configuration file.
 *
 * - `react()` enables JSX/TSX compilation and React Fast Refresh (HMR).
 * - `tailwindcss()` processes Tailwind utility classes at build time.
 */
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Backend routes live at the root (/pipelines, /items, /plugins),
      // so the /api prefix is stripped — mirrors the production nginx.
      "/api": {
        target: "http://localhost:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // MLflow runs with --static-prefix /mlflow, so the prefix is kept.
      "/mlflow": "http://localhost:5000",
    },
  },
})
