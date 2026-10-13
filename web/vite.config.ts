import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 6000,
    rollupOptions: { input: { app: 'index.html', verification: 'verify.html', comparison: 'compare.html', guiBenchmark:'gui-benchmark.html' } } },
});
