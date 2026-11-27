import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    // Mode pengembangan: `/api` diteruskan ke FastAPI (uvicorn api.main:app --port 8000).
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false } },
  },
})
