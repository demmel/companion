import { useMemo } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import { useTriggerEvents } from "./useTriggerEvents";
import { useTimelineHistory } from "./useTimelineHistory";
import { AgentClient } from "../client";
import { TimelineEntry, TimelineEntryTrigger, ContextInfo } from "../types";

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
}

export function useTimeline(
  client: AgentClient,
  events: ClientAgentEvent[],
): UseTimelineReturn {
  // Historical entries from API
  const historyData = useTimelineHistory(client);

  // Streaming entries from WebSocket events
  const streamingData = useTriggerEvents(events);

  // Combine historical + streaming entries
  const combinedEntries = useMemo(() => {
    // Convert streaming trigger entries to timeline entries
    const streamingTimelineEntries: TimelineEntry[] =
      streamingData.streamingEntries.map((entry) => ({
        type: "trigger" as const,
        entry,
      }));

    const combined = [...historyData.entries, ...streamingTimelineEntries];

    // Debug: check for duplicates by extracting entry_id from TimelineEntry
    const entryIds = combined
      .filter((e): e is TimelineEntryTrigger => e.type === "trigger")
      .map((e) => e.entry.entry_id);
    const duplicates = entryIds.filter(
      (id, index) => entryIds.indexOf(id) !== index,
    );
    if (duplicates.length > 0) {
      console.warn("Duplicate entry IDs found:", duplicates);
      console.log(
        "Historical trigger entries:",
        historyData.entries
          .filter((e): e is TimelineEntryTrigger => e.type === "trigger")
          .map((e) => e.entry.entry_id),
      );
      console.log(
        "Streaming entries:",
        streamingData.streamingEntries.map((e) => e.entry_id),
      );
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
