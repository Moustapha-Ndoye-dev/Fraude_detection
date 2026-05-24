import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production")
  },
  build: {
    emptyOutDir: true,
    minify: true,
    outDir: "dist",
    lib: {
      entry: "src/index.jsx",
      formats: ["es"],
      fileName: "fraud-widgets"
    }
  }
});
