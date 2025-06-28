import { useState, useRef, useCallback, useEffect } from 'react';
import { AgentEvent } from '../types';

export interface UseStreamBatcherReturn {
  events: AgentEvent[];
  queueEvent: (event: AgentEvent) => void;
  clearEvents: () => void;
}

export function useStreamBatcher(batchInterval: number = 50): UseStreamBatcherReturn {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const eventBufferRef = useRef<AgentEvent[]>([]);
  const batchTimerRef = useRef<number | undefined>(undefined);

  const processBatch = useCallback(() => {
    if (eventBufferRef.current.length === 0) return;
    
    const newEvents = eventBufferRef.current.splice(0); // Clear buffer
    setEvents(prevEvents => [...prevEvents, ...newEvents]);
  }, []);

  const queueEvent = useCallback((event: AgentEvent) => {
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
  }, [processBatch, batchInterval]);

  const clearEvents = useCallback(() => {
    eventBufferRef.current = [];
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
    }
    setEvents([]);
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
    clearEvents
  };
}