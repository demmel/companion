import { useState, useCallback, useEffect } from "react";
import { AgentClient } from "../client";
import { TimelineEntry, TimelineResponse, PaginationInfo } from "../types";

export interface UseTimelineHistoryReturn {
  // Historical data
  entries: TimelineEntry[];

  // Pagination state
  canLoadMore: boolean;
  isLoadingMore: boolean;
  hasLoadedAnyData: boolean;

  // Actions
  loadInitialData: () => Promise<void>;
  loadMore: () => Promise<void>;
  clear: () => void;
}

export function useTimelineHistory(
  client: AgentClient,
  hydrationEntries: TimelineEntry[],
  hydrationPagination: PaginationInfo | null,
): UseTimelineHistoryReturn {
  const [entries, setEntries] = useState<TimelineEntry[]>([]);
  const [previousCursor, setPreviousCursor] = useState<string | null>(null);
  const [canLoadMore, setCanLoadMore] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasLoadedAnyData, setHasLoadedAnyData] = useState(false);

  // Initialize from hydration data
  useEffect(() => {
    if (hydrationEntries.length > 0 && !hasLoadedAnyData) {
      setEntries(hydrationEntries);
      if (hydrationPagination) {
        setPreviousCursor(hydrationPagination.previous_cursor || null);
        setCanLoadMore(hydrationPagination.has_previous);
      }
      setHasLoadedAnyData(true);
    }
  }, [hydrationEntries, hydrationPagination, hasLoadedAnyData]);

  const processTimelineResponse = useCallback(
    (response: TimelineResponse, isPrepend: boolean = false) => {
      // Use all timeline entries (triggers and summaries)
      const newEntries = response.entries;

      // Update entries
      if (isPrepend) {
        // Prepend older data for "load more"
        setEntries((prev) => [...newEntries, ...prev]);
      } else {
        // Replace for initial load
        setEntries(newEntries);
      }

      // Update pagination state
      setPreviousCursor(response.pagination.previous_cursor || null);
      setCanLoadMore(response.pagination.has_previous);
    },
    [],
  );

  const loadInitialData = useCallback(async () => {
    if (isLoadingMore) return;

    setIsLoadingMore(true);
    try {
      // Load most recent page (default API behavior)
      const response = await client.getTimeline(3);
      processTimelineResponse(response, false);
      setHasLoadedAnyData(true);
    } catch (error) {
      console.error("Failed to load initial timeline:", error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [client, processTimelineResponse]);

  const loadMore = useCallback(async () => {
    if (isLoadingMore || !canLoadMore || !previousCursor) return;

    setIsLoadingMore(true);
    try {
      // Load previous page (older entries)
      const response = await client.getTimeline(5, previousCursor);
      processTimelineResponse(response, true); // prepend older data
    } catch (error) {
      console.error("Failed to load more timeline data:", error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [client, previousCursor, processTimelineResponse]);

  const clear = useCallback(() => {
    setEntries([]);
    setPreviousCursor(null);
    setCanLoadMore(false);
    setHasLoadedAnyData(false);
  }, []);

  return {
    entries,
    canLoadMore,
    isLoadingMore,
    hasLoadedAnyData,
    loadInitialData,
    loadMore,
    clear,
  };
}
