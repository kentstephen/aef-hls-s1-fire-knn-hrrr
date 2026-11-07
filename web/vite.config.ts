import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

/**
 * Development only: embedding chunks kept on disk by scripts/replay_download.py (data/replay/<fire>/<year>/<cy>_<cx>.bin)
 * served at /local-aef/<year>/<cy>_<cx>.bin; the chunk worker reads there first and falls back to the mosaic on a 404.
 */
function localAef(): Plugin {
  const root = fileURLToPath(new URL("../data/replay", import.meta.url));
  return {
    name: "local-aef",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/local-aef", (req, res, next) => {
        const m = req.url?.match(/^\/(\d{4})\/(\d+)_(\d+)\.bin$/);
        if (!m) return next();
        const fires = fs.existsSync(root) ? fs.readdirSync(root) : [];
        for (const fire of fires) {
          const f = path.join(root, fire, m[1]!, `${m[2]}_${m[3]}.bin`);
          if (fs.existsSync(f)) {
            res.setHeader("Content-Type", "application/octet-stream");
            // read once into the page; the browser's HTTP cache would only hold a second copy
            res.setHeader("Cache-Control", "no-store");
            fs.createReadStream(f).pipe(res);
            return;
          }
        }
        res.statusCode = 404;
        res.end();
      });
    },
  };
}

export default defineConfig(({ command }) => ({
  plugins: [react(), localAef()],
  // Project Pages site: production asset URLs need the repo prefix; local dev stays at "/".
  base: command === "build" ? "/aef-hls-s1-fire-knn-hrrr/" : "/",
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
