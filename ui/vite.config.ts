import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: { outDir: "../byosm/app/static", emptyOutDir: true },
  server: { proxy: { "/api": { target: "http://127.0.0.1:8765", ws: true } } },
})
