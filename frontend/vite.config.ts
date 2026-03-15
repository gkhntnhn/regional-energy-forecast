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
      "/admin": "http://localhost:8000",
    },
  },
  build: {
    outDir: "../src/energy_forecast/serving/static/dist",
    emptyOutDir: true,
  },
});
