import { useState, useCallback, useEffect, useMemo } from "react";
import { ClientAgentEvent, useWebSocket } from "@/hooks/useWebSocket";
import { useStreamBatcher } from "@/hooks/useStreamBatcher";
import { useTriggerEvents } from "@/hooks/useTriggerEvents";
import { useSmartScroll } from "@/hooks/useSmartScroll";
import { ChatHeader } from "@/components/ChatHeader";
import { ChatInput } from "@/components/ChatInput";
import { Timeline } from "@/components/Timeline";
import { AgentClient } from "@/client";
import { css } from "@styled-system/css";
import { debug } from "@/utils/debug";

interface ChatInterfaceProps {
  client: AgentClient;
}

export function ChatInterface({ client }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState("");

  // New architecture: batch events then convert to structured messages
  const { events, queueEvent, clearEvents } = useStreamBatcher(50);

  useMemo(() => {
    debug.log("Events:", events);
  }, [events]);

  const {
    triggerEntries,
    summaries,
    isStreamActive,
    contextInfo,
    setContextInfo,
    loadTriggerHistory,
    clearTriggerHistory,
  } = useTriggerEvents(events);

  const handleMessage = useCallback(
    (event: ClientAgentEvent) => {
      queueEvent(event);
    },
    [queueEvent],
  );

  const handleError = useCallback((error: Event) => {
    console.error("WebSocket error:", error);
  }, []);

  const { isConnected, isConnecting, sendMessage } = useWebSocket({
    url: client.chatWsUrl,
    onMessage: handleMessage,
    onError: handleError,
  });

  const {
    messagesEndRef,
    messagesContainerRef,
    handleScroll,
    setUserAtBottom,
  } = useSmartScroll({
    items: triggerEntries,
  });

  const handleSubmit = (message: string) => {
    // Send to server
    sendMessage(message);
    setInputValue("");

    // When user sends a message, they probably want to see the response
    setUserAtBottom(true);
  };

  // Load initial data on mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Load trigger history and initial context info in parallel
        const [triggerData, contextData] = await Promise.all([
          client.getTriggerHistory(),
          client.getContextInfo(),
        ]);

        if (triggerData.entries.length > 0 || triggerData.summaries.length > 0) {
          loadTriggerHistory(triggerData.entries, triggerData.summaries);

          // Set initial scroll position to bottom instantly
          requestAnimationFrame(() => {
            if (messagesEndRef.current) {
              messagesEndRef.current.scrollIntoView({ behavior: "instant" });
            }
          });
        }
        if (contextData) {
          setContextInfo(contextData);
        }
      } catch (error) {
        console.error("Failed to load initial data:", error);
      }
    };

    loadInitialData();
  }, [client, loadTriggerHistory]);

  const handleClear = async () => {
    try {
      await client.reset();
    } catch (error) {
      console.error("Error resetting server:", error);
    }

    clearEvents();
    clearTriggerHistory();
    setUserAtBottom(true);
  };

  return (
    <div
      className={css({
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        bg: "gray.900",
      })}
    >
      <ChatHeader isConnected={isConnected} isConnecting={isConnecting} />

      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
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
          <Timeline
            triggerEntries={triggerEntries}
            summaries={summaries}
            isStreamActive={isStreamActive}
          />
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
      />
    </div>
  );
}
