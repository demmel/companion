import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useAgentWebSocket } from "@/hooks/useAgentWebSocket";
import { useStreamBatcher } from "@/hooks/useStreamBatcher";
import { useTimeline } from "@/hooks/useTimeline";
import { useUsername } from "@/contexts/UsernameContext";
import { ChatHeader } from "@/components/ChatHeader";
import { ChatInput } from "@/components/ChatInput";
import { Timeline } from "@/components/Timeline";
import { AgentClient } from "@/client";
import { TimelineEntry, PaginationInfo } from "@/types";
import { css } from "@styled-system/css";
import { debug } from "@/utils/debug";

interface ChatInterfaceProps {
  client: AgentClient;
}

export function ChatInterface({ client }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState("");
  const { username } = useUsername();

  // Hydration state (from WebSocket)
  const [timelineEntries, setTimelineEntries] = useState<TimelineEntry[]>([]);
  const [paginationInfo, setPaginationInfo] = useState<PaginationInfo | null>(
    null,
  );

  const handleError = useCallback((error: Event) => {
    console.error("WebSocket error:", error);
  }, []);

  const handleHydration = useCallback(
    (entries: TimelineEntry[], pagination: PaginationInfo) => {
      setTimelineEntries(entries);
      setPaginationInfo(pagination);
    },
    [],
  );

  // useAgentWebSocket handles protocol discrimination
  const { agentEvents, isConnected, isConnecting, sendMessage } =
    useAgentWebSocket({
      url: client.chatWsUrl,
      onError: handleError,
      onHydration: handleHydration,
    });

  // Batch streaming events
  const { events, queueEvent, clearEvents, orphanedEventCount } =
    useStreamBatcher(50);

  // Queue agent events as they arrive
  useEffect(() => {
    agentEvents.forEach((event) => queueEvent(event));
  }, [agentEvents, queueEvent]);

  useMemo(() => {
    debug.log("Events:", events);
  }, [events]);

  const {
    triggerEntries,
    isStreamActive,
    contextInfo,
    setContextInfo,
    canLoadMore,
    isLoadingMore,
    loadMore,
    clearData,
  } = useTimeline(client, events, timelineEntries, paginationInfo);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (message: string, imageIds?: string[]) => {
    // Send to server with username
    sendMessage(message, username, imageIds);
    setInputValue("");

    // Scroll to bottom after sending message
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }, 100);
  };

  // Load context info on mount (timeline comes from hydration now)
  useEffect(() => {
    const loadData = async () => {
      try {
        const contextData = await client.getContextInfo();
        if (contextData) {
          setContextInfo(contextData);
        }
      } catch (error) {
        console.error("Failed to load context info:", error);
      }
    };

    loadData();
  }, [client, setContextInfo]);

  // Scroll to bottom when entries are first loaded
  const hasLoadedInitialData = useRef(false);
  useEffect(() => {
    if (triggerEntries.length > 0 && !hasLoadedInitialData.current) {
      hasLoadedInitialData.current = true;
      // Wait for DOM to update, then scroll
      requestAnimationFrame(() => {
        if (messagesEndRef.current) {
          messagesEndRef.current.scrollIntoView({ behavior: "instant" });
        }
      });
    }
  }, [triggerEntries]);

  const handleClear = async () => {
    try {
      await client.reset();
    } catch (error) {
      console.error("Error resetting server:", error);
    }

    // Clear all client state
    clearEvents();
    clearData();
    setTimelineEntries([]);
    setPaginationInfo(null);
  };

  return (
    <div
      className={css({
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        flexDirection: "column",
        bg: "gray.900",
      })}
    >
      <ChatHeader
        isConnected={isConnected}
        isConnecting={isConnecting}
        client={client}
      />

      {/* Show orphaned event notification */}
      {orphanedEventCount > 0 && (
        <div
          className={css({
            bg: "orange.700",
            borderBottom: "1px solid",
            borderColor: "orange.600",
            px: 4,
            py: 2,
            fontSize: "sm",
            color: "orange.100",
          })}
        >
          ⚠️ Connected mid-stream: {orphanedEventCount} event
          {orphanedEventCount === 1 ? "" : "s"} skipped to maintain consistency
        </div>
      )}

      <div
        className={css({
          flex: 1,
          overflowY: "auto",
          px: 4,
          py: 4,
        })}
      >
        {triggerEntries.length === 0 && (
          <div
            className={css({
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "full",
              textAlign: "center",
            })}
          >
            <div className={css({ maxWidth: "md" })}>
              <div
                className={css({
                  fontSize: "4xl",
                  mb: 6,
                  opacity: 0.5,
                })}
              >
                💬
              </div>
              <h2
                className={css({
                  fontSize: "xl",
                  fontWeight: "medium",
                  color: "gray.300",
                  mb: 2,
                })}
              >
                Start a conversation
              </h2>
              <p
                className={css({
                  color: "gray.500",
                  fontSize: "xl",
                  mb: 6,
                })}
              >
                Send a message to begin chatting with the agent
              </p>
              <div
                className={css({
                  bg: "gray.800",
                  border: "1px solid",
                  borderColor: "gray.700",
                  rounded: "lg",
                  p: 4,
                  textAlign: "left",
                })}
              >
                <p
                  className={css({
                    fontSize: "xs",
                    color: "gray.400",
                    textTransform: "uppercase",
                    letterSpacing: "wide",
                    mb: 2,
                  })}
                >
                  Example
                </p>
                <p
                  className={css({
                    color: "gray.300",
                    fontSize: "xl",
                  })}
                >
                  "Please roleplay as Elena, a mysterious vampire."
                </p>
              </div>
            </div>
          </div>
        )}

        {triggerEntries.length > 0 && (
          <>
            {/* Load Previous button */}
            {canLoadMore && (
              <div
                className={css({
                  display: "flex",
                  justifyContent: "center",
                  mb: 4,
                })}
              >
                <button
                  onClick={loadMore}
                  disabled={isLoadingMore}
                  className={css({
                    px: 4,
                    py: 2,
                    bg: "blue.600",
                    color: "white",
                    rounded: "md",
                    fontSize: "sm",
                    fontWeight: "medium",
                    transition: "all 0.2s",
                    cursor: isLoadingMore ? "not-allowed" : "pointer",
                    opacity: isLoadingMore ? 0.5 : 1,
                    _hover: {
                      bg: isLoadingMore ? "blue.600" : "blue.700",
                    },
                  })}
                >
                  {isLoadingMore ? "Loading..." : "Load Previous"}
                </button>
              </div>
            )}

            <Timeline
              entries={triggerEntries}
              isStreamActive={isStreamActive}
            />
          </>
        )}

        <div ref={messagesEndRef} />
      </div>

      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        disabled={!isConnected || isConnecting || isStreamActive}
        onClear={handleClear}
        clearDisabled={isStreamActive}
        contextInfo={contextInfo}
        allowEmptySubmit={triggerEntries.length > 0}
        client={client}
      />
    </div>
  );
}
