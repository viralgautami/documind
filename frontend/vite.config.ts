import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_DEMO === 'true' ? '/documind/' : '/',
  server: { proxy: { '/api': 'http://127.0.0.1:5000' } },
})
