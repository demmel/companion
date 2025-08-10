import { useMemo } from "react";
import { ChatInterface } from "./components/ChatInterface";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { AgentClient } from "./client";

function App() {
  const client = useMemo(() => {
    // In development, use env var to point to agent server
    // In production, use relative URLs (same origin as client)
    const isDev = import.meta.env.DEV;

    if (isDev) {
      const host = import.meta.env.VITE_AGENT_HOST || "localhost";
      const port = parseInt(import.meta.env.VITE_AGENT_PORT || "8000");
      const client = new AgentClient({ host, port });

      // Log configuration in development for debugging
      console.log(`[DEV] Agent client connecting to: ${client.httpBaseUrl}`);

      return client;
    } else {
      // Production: assume client is served from agent server
      const { hostname, port } = window.location;
      return new AgentClient({
        host: hostname,
        port: parseInt(port) || 80,
      });
    }
  }, []);

  return (
    <ErrorBoundary>
      <ChatInterface client={client} />
    </ErrorBoundary>
  );
}

export default App;
