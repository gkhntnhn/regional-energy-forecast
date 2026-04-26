import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/predict": "http://localhost:8000",
      "/status": "http://localhost:8000",
      "/models": "http://localhost:8000",
      "/jobs": "http://localhost:8000",
      "/files": "http://localhost:8000",
      "/admin/analytics": "http://localhost:8000",
      "/admin/jobs": "http://localhost:8000",
      "/admin/models": "http://localhost:8000",
      "/admin/system": "http://localhost:8000",
    },
  },
  build: {
    outDir: "../src/energy_forecast/serving/static/dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // Heavy chart lib — split so main bundle stays under 500 kB warning.
          // Loaded lazily via HistoryPage / AdminPage code-split.
          recharts: ["recharts"],
          // React core stays together — shared across every page.
          "react-vendor": ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
});
