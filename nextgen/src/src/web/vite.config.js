import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build ra thẳng app/static để FastAPI phục vụ (gộp 1 nơi trên HF Spaces).
// Dev: proxy /ask, /health sang backend uvicorn cổng 8000.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../app/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/ask": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/config": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
