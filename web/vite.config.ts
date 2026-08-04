import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const BACKEND_PROXY_PATHS = [
  '/health',
  '/entities',
  '/entities-text',
  '/detect',
  '/redact',
  '/detect-text',
  '/redact-text',
]

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      BACKEND_PROXY_PATHS.map((path) => [path, 'http://localhost:7860']),
    ),
  },
})
