/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@styled-system': resolve(__dirname, './styled-system'),
    },
  },
  server: {
    proxy: {
      '/generated_images': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/generated_audio': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/uploaded_images': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
