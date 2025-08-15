import { useMemo } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import { useTriggerEvents } from "./useTriggerEvents";
import { useTimelineHistory } from "./useTimelineHistory";
import { AgentClient } from "../client";
import { TriggerHistoryEntry, ContextInfo } from "../types";

export interface UseTimelineReturn {
  // Combined timeline data
  triggerEntries: TriggerHistoryEntry[];
  
  // Stream state
  isStreamActive: boolean;
  contextInfo: ContextInfo | null;
  setContextInfo: (context: ContextInfo) => void;
  
  // Pagination
  canLoadMore: boolean;
  isLoadingMore: boolean;
  hasLoadedAnyData: boolean;
  loadMore: () => Promise<void>;
  
  // Actions
  loadInitialData: () => Promise<void>;
  clearData: () => void;
}

export function useTimeline(client: AgentClient, events: ClientAgentEvent[]): UseTimelineReturn {
  // Historical entries from API
  const historyData = useTimelineHistory(client);
  
  // Streaming entries from WebSocket events
  const streamingData = useTriggerEvents(events);
  
  // Combine historical + streaming entries
  const combinedEntries = useMemo(() => {
    const combined = [...historyData.entries, ...streamingData.streamingEntries];
    
    // Debug: check for duplicates
    const entryIds = combined.map(e => e.entry_id);
    const duplicates = entryIds.filter((id, index) => entryIds.indexOf(id) !== index);
    if (duplicates.length > 0) {
      console.warn('Duplicate entry IDs found:', duplicates);
      console.log('Historical entries:', historyData.entries.map(e => e.entry_id));
      console.log('Streaming entries:', streamingData.streamingEntries.map(e => e.entry_id));
    }
    
    return combined;
  }, [historyData.entries, streamingData.streamingEntries]);

  const clearData = () => {
    historyData.clear();
    streamingData.clearStreamingData();
  };

  return {
    // Combined data
    triggerEntries: combinedEntries,
    
    // Stream state (from streaming hook)
    isStreamActive: streamingData.isStreamActive,
    contextInfo: streamingData.contextInfo,
    setContextInfo: streamingData.setContextInfo,
    
    // Pagination (from history hook)
    canLoadMore: historyData.canLoadMore,
    isLoadingMore: historyData.isLoadingMore, 
    hasLoadedAnyData: historyData.hasLoadedAnyData,
    loadMore: historyData.loadMore,
    
    // Actions
    loadInitialData: historyData.loadInitialData,
    clearData,
  };
}