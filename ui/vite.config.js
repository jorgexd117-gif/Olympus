import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        host: "0.0.0.0",
        port: 5000,
        allowedHosts: true,
        proxy: {
            "/health": { target: "http://localhost:8000", changeOrigin: true },
            "/healthz": { target: "http://localhost:8000", changeOrigin: true },
            "/projects": { target: "http://localhost:8000", changeOrigin: true },
            "/profiles": { target: "http://localhost:8000", changeOrigin: true },
            "/agent": { target: "http://localhost:8000", changeOrigin: true },
            "/agents": { target: "http://localhost:8000", changeOrigin: true },
            "/assistant": { target: "http://localhost:8000", changeOrigin: true },
            "/commands": { target: "http://localhost:8000", changeOrigin: true },
            "/orchestrator": { target: "http://localhost:8000", changeOrigin: true },
            "/memories": { target: "http://localhost:8000", changeOrigin: true },
            "/conversations": { target: "http://localhost:8000", changeOrigin: true },
            "/models": { target: "http://localhost:8000", changeOrigin: true },
            "/api": { target: "http://localhost:8000", changeOrigin: true },
        },
    },
});
