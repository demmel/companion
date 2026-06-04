import { useMemo, useCallback } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import { useTriggerEvents } from "./useTriggerEvents";
import { useTimelineHistory } from "./useTimelineHistory";
import { AgentClient } from "../client";
import {
  TimelineEntry,
  TimelineEntryTrigger,
  ContextInfo,
  PaginationInfo,
  Action,
} from "../types";

export interface UseTimelineReturn {
  // Combined timeline data
  triggerEntries: TimelineEntry[];

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
  updateAction: (
    triggerId: string,
    actionIndex: number,
    updates: Partial<Action>,
  ) => void;
}

export function useTimeline(
  client: AgentClient,
  events: ClientAgentEvent[],
  hydrationEntries: TimelineEntry[],
  hydrationPagination: PaginationInfo | null,
): UseTimelineReturn {
  // Historical entries from hydration or REST API
  const historyData = useTimelineHistory(
    client,
    hydrationEntries,
    hydrationPagination,
  );

  // Streaming entries from WebSocket events
  const streamingData = useTriggerEvents(events);

  // Combine historical + streaming entries
  const combinedEntries = useMemo(() => {
    // Get entry IDs from historical data to filter out duplicates
    const historicalEntryIds = new Set(
      historyData.entries
        .filter((e): e is TimelineEntryTrigger => e.type === "trigger")
        .map((e) => e.entry.entry_id),
    );

    // Filter streaming entries to exclude those already in history
    const uniqueStreamingEntries = streamingData.streamingEntries.filter(
      (entry) => !historicalEntryIds.has(entry.entry_id),
    );

    // Convert streaming trigger entries to timeline entries
    const streamingTimelineEntries: TimelineEntry[] =
      uniqueStreamingEntries.map((entry) => ({
        type: "trigger" as const,
        entry,
      }));

    // Combine and sort by timestamp
    const combined = [...historyData.entries, ...streamingTimelineEntries];
    combined.sort((a, b) => {
      if (a.type === "summary" || b.type === "summary") {
        return 0;
      }
      const aTime = new Date(a.entry.timestamp).getTime();
      const bTime = new Date(b.entry.timestamp).getTime();
      return aTime - bTime;
    });

    return combined;
  }, [historyData.entries, streamingData.streamingEntries]);

  const clearData = () => {
    historyData.clear();
    streamingData.clearStreamingData();
  };

  const updateAction = useCallback(
    (triggerId: string, actionIndex: number, updates: Partial<Action>) => {
      // Update both streaming and historical data
      streamingData.updateAction(triggerId, actionIndex, updates);
      historyData.updateAction(triggerId, actionIndex, updates);
    },
    [streamingData, historyData],
  );

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
    updateAction,
  };
}
