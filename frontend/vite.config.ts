/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": process.env.VITE_API_PROXY ?? "http://localhost:8000" } },
  preview: { port: 4173, proxy: { "/api": process.env.VITE_API_PROXY ?? "http://localhost:8000" } },
  build: {
    chunkSizeWarningLimit: 5000,
    rollupOptions: { output: { manualChunks: { plotly: ["plotly.js-dist-min", "react-plotly.js"], maplibre: ["maplibre-gl"] } } },
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});
