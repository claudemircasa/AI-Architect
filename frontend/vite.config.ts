/**
 * > [!AML-DOC-FILE]
 * @file        vite.config.ts
 * @description Vite build configuration for the editor frontend.
 * @module      frontend/config
 * @exports     default config
 * @created     2026-09-30
 * @context     Port 5173 is allow-listed by the backend's CORS policy [amm: E.1],
 *              and `strictPort` keeps it from silently moving to 5174, which would
 *              make requests fail CORS instead of failing to bind.
 */

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true, target: "safari16" },
  clearScreen: false,
});
