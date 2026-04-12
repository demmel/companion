import { useEffect, useMemo, useState } from "react";
import { ChatInterface } from "./components/ChatInterface";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { OllamaModelsPage } from "./components/OllamaModelsPage";
import { UsernameProvider } from "./contexts/UsernameContext";
import { AgentClient } from "./client";

type AppRoute = "chat" | "ollama-models";

function getRouteFromPath(pathname: string): AppRoute {
  return pathname.startsWith("/ollama-models") ? "ollama-models" : "chat";
}

function App() {
  const client = useMemo(() => {
    // In development, use env var to point to agent server
    // In production, use relative URLs (same origin as client)
    const isDev = import.meta.env.DEV;

    if (isDev) {
      const host = import.meta.env.VITE_AGENT_HOST || "localhost";
      const port = parseInt(import.meta.env.VITE_AGENT_PORT || "8080");
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
  const [route, setRoute] = useState<AppRoute>(() =>
    getRouteFromPath(window.location.pathname),
  );

  useEffect(() => {
    const handlePopState = () => {
      setRoute(getRouteFromPath(window.location.pathname));
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigateTo = (nextRoute: AppRoute) => {
    const nextPath = nextRoute === "ollama-models" ? "/ollama-models" : "/";
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setRoute(nextRoute);
  };

  return (
    <ErrorBoundary>
      <UsernameProvider>
        {route === "ollama-models" ? (
          <OllamaModelsPage client={client} onBack={() => navigateTo("chat")} />
        ) : (
          <ChatInterface
            client={client}
            onNavigateToOllamaModels={() => navigateTo("ollama-models")}
          />
        )}
      </UsernameProvider>
    </ErrorBoundary>
  );
}

export default App;
