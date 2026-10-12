import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The FastAPI backend runs on :8000. Proxying keeps the browser on one origin in development.
    // 127.0.0.1 rather than localhost: uvicorn listens on IPv4, and localhost can resolve to IPv6 first.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
