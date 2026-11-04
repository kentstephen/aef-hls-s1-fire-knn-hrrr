import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Project Pages site: production asset URLs need the repo prefix; local dev stays at "/".
  base: command === "build" ? "/aef-hls-s1-fire-knn/" : "/",
  worker: { format: "es" },
  server: {
    // Override with PORT; strictPort fails loudly.
    port: Number(process.env.PORT) || 5374,
    strictPort: true,
  },
  optimizeDeps: {
    exclude: ["lerc"],
  },
}));
