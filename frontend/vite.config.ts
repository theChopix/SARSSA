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
})
