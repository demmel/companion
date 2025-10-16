import { useState, useRef, useCallback, useEffect } from "react";
import { ClientAgentEvent } from "./useWebSocket";

export interface UseStreamBatcherReturn {
  events: ClientAgentEvent[];
  queueEvent: (event: ClientAgentEvent) => void;
  clearEvents: () => void;
  orphanedEventCount: number;
}

export function useStreamBatcher(
  batchInterval: number = 50,
): UseStreamBatcherReturn {
  const [events, setEvents] = useState<ClientAgentEvent[]>([]);
  const [orphanedEventCount, setOrphanedEventCount] = useState(0);
  const eventBufferRef = useRef<ClientAgentEvent[]>([]);
  const batchTimerRef = useRef<number | undefined>(undefined);

  // Track active triggers to filter orphaned events
  const activeTriggerIds = useRef<Set<string>>(new Set());

  const processBatch = useCallback(() => {
    if (eventBufferRef.current.length === 0) return;

    const newEvents = eventBufferRef.current.splice(0); // Clear buffer
    setEvents((prevEvents) => [...prevEvents, ...newEvents]);
  }, []);

  const queueEvent = useCallback(
    (event: ClientAgentEvent) => {
      // Filter orphaned events - only accept events for triggers we know about
      const shouldAcceptEvent = (() => {
        switch (event.type) {
          case "trigger_started":
            // Always accept trigger_started - this begins a new sequence
            activeTriggerIds.current.add(event.entry_id);
            return true;

          case "trigger_completed":
            // Accept if we have this trigger, then clean it up
            const hasCompletedTrigger = activeTriggerIds.current.has(
              event.entry.entry_id,
            );
            activeTriggerIds.current.delete(event.entry.entry_id);
            return hasCompletedTrigger;

          case "action_started":
          case "action_progress":
          case "action_completed":
            // Accept if we have this trigger
            return activeTriggerIds.current.has(event.entry_id);

          case "summarization_started":
          case "summarization_finished":
          case "error":
            // Global events - always accept
            return true;

          default:
            // Unknown event type - accept to be safe
            return true;
        }
      })();

      if (!shouldAcceptEvent) {
        // Count orphaned event but don't process it
        setOrphanedEventCount((count) => count + 1);
        return;
      }

      eventBufferRef.current.push(event);

      // Reset batch timer
      if (batchTimerRef.current) {
        clearTimeout(batchTimerRef.current);
      }

      if (batchInterval === 0) {
        // Process immediately
        processBatch();
      } else {
        batchTimerRef.current = window.setTimeout(processBatch, batchInterval);
      }
    },
    [processBatch, batchInterval],
  );

  const clearEvents = useCallback(() => {
    eventBufferRef.current = [];
    activeTriggerIds.current.clear();
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
    }
    setEvents([]);
    setOrphanedEventCount(0);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (batchTimerRef.current) {
        clearTimeout(batchTimerRef.current);
      }
    };
  }, []);

  return {
    events,
    queueEvent,
    clearEvents,
    orphanedEventCount,
  };
}
