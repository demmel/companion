import { AgentEvent } from "@/agent_events";
import { useState, useEffect, useRef, useCallback } from "react";

export type ClientAgentEvent = {
  id: number;
} & AgentEvent;

export interface UseWebSocketOptions {
  url: string;
  onMessage?: (event: ClientAgentEvent) => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  sendMessage: (message: string, username: string, imageIds?: string[]) => void;
  disconnect: () => void;
  connect: () => void;
}

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  reconnectAttempts = 3,
  reconnectDelay = 1000,
}: UseWebSocketOptions): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const currentId = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        reconnectCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (!data || typeof data !== "object" || !data.type) {
            console.warn("Received invalid WebSocket message:", data);
            return;
          }
          const agent_event: ClientAgentEvent = {
            ...data,
            id: currentId.current++,
          };
          onMessage?.(agent_event);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);

        // Attempt to reconnect
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, reconnectDelay * reconnectCountRef.current);
        }
      };

      ws.onerror = (error) => {
        setIsConnecting(false);
        onError?.(error);
      };
    } catch (error) {
      setIsConnecting(false);
      console.error("Failed to create WebSocket connection:", error);
    }
  }, [url, onMessage, onError, reconnectAttempts, reconnectDelay]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
    reconnectCountRef.current = 0;
  }, []);

  const sendMessage = useCallback((message: string, username: string, imageIds?: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const payload: { message: string; image_ids?: string[]; user_name?: string } = { message };
      if (imageIds && imageIds.length > 0) {
        payload.image_ids = imageIds;
      }
      payload.user_name = username;
      wsRef.current.send(JSON.stringify(payload));
    } else {
      console.warn("WebSocket is not connected");
    }
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    sendMessage,
    disconnect,
    connect,
  };
};
