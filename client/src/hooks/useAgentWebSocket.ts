import { useState, useRef, useCallback, useEffect } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import {
  AgentServerEvent,
  HydrationResponse,
  EventEnvelope,
} from "@/agent_events";
import { TimelineEntry, PaginationInfo } from "@/types";

export interface UseAgentWebSocketOptions {
  url: string;
  onError?: (error: Event) => void;
  onHydration?: (entries: TimelineEntry[], pagination: PaginationInfo) => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export interface UseAgentWebSocketReturn {
  // Unwrapped data from server
  agentEvents: ClientAgentEvent[];

  // WebSocket state
  isConnected: boolean;
  isConnecting: boolean;

  // Actions
  sendMessage: (message: string, username: string, imageIds?: string[]) => void;
  disconnect: () => void;
  connect: () => void;
}

/**
 * Agent-aware WebSocket hook that handles protocol discrimination.
 * Manages its own WebSocket connection and:
 * - Sends hydration request on connect
 * - Discriminates AgentServerEvent (HydrationResponse | EventEnvelope)
 * - Unwraps EventEnvelope to AgentEvent
 * - Tracks last (trigger_id, sequence) for reconnection
 */
export const useAgentWebSocket = ({
  url,
  onError,
  onHydration,
  reconnectAttempts = 3,
  reconnectDelay = 1000,
}: UseAgentWebSocketOptions): UseAgentWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [agentEvents, setAgentEvents] = useState<ClientAgentEvent[]>([]);

  // Track last known state for reconnection
  const lastTriggerId = useRef<string | null>(null);
  const lastEventSequence = useRef<number | null>(null);
  const eventIdCounter = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);

  // Handle discrimination and unwrapping
  const handleServerEvent = useCallback(
    (rawData: string) => {
      try {
        const serverEvent = JSON.parse(rawData) as AgentServerEvent;

        switch (serverEvent.type) {
          case "hydration_response": {
            const hydration = serverEvent as HydrationResponse;
            console.log(
              `[Hydration] Received ${hydration.entries.length} timeline entries`,
            );

            // Call hydration callback if provided
            onHydration?.(hydration.entries, hydration.pagination);
            break;
          }

          case "event_envelope": {
            const envelope = serverEvent as EventEnvelope;

            // Track last known state for reconnection
            lastTriggerId.current = envelope.trigger_id;
            lastEventSequence.current = envelope.event_sequence;

            // Unwrap to AgentEvent and add client ID
            const clientEvent: ClientAgentEvent = {
              ...envelope.event,
              id: eventIdCounter.current++,
            };

            setAgentEvents((prev) => [...prev, clientEvent]);
            break;
          }

          default: {
            const exhaustiveCheck: never = serverEvent;
            console.warn("Unknown server event type:", exhaustiveCheck);
          }
        }
      } catch (error) {
        console.error("Failed to parse server event:", error);
      }
    },
    [onHydration],
  );

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

        // Send hydration request
        const hydrationRequest = {
          type: "hydrate",
          last_trigger_id: lastTriggerId.current,
          last_event_sequence: lastEventSequence.current,
        };

        console.log(
          `[Hydration] Sending request:`,
          hydrationRequest.last_trigger_id
            ? `from trigger ${hydrationRequest.last_trigger_id} seq ${hydrationRequest.last_event_sequence}`
            : "fresh connection (last 3 triggers)",
        );

        ws.send(JSON.stringify(hydrationRequest));
      };

      ws.onmessage = (event) => {
        handleServerEvent(event.data);
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
  }, [url, handleServerEvent, onError, reconnectAttempts, reconnectDelay]);

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

  const sendMessage = useCallback(
    (message: string, username: string, imageIds?: string[]) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const payload: {
          type: "message";
          message: string;
          image_ids?: string[];
          user_name?: string;
        } = { type: "message", message };
        if (imageIds && imageIds.length > 0) {
          payload.image_ids = imageIds;
        }
        payload.user_name = username;
        wsRef.current.send(JSON.stringify(payload));
      } else {
        console.warn("WebSocket is not connected");
      }
    },
    [],
  );

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    agentEvents,
    isConnected,
    isConnecting,
    sendMessage,
    disconnect,
    connect,
  };
};
