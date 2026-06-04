import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import {
  TriggerHistoryEntry,
  Action,
  ActionStreamingData,
  Trigger,
  ContextInfo,
} from "../types";
import { debug } from "@/utils/debug";

interface ActionBuilder {
  action: Action;
  sequence_number: number;
  action_number: number;
  partial_results: string[];
}

// Single active trigger builder (only one trigger can be active at a time)
interface ActiveTriggerBuilder {
  entry_id: string;
  trigger: Trigger;
  actions: ActionBuilder[]; // Array to maintain execution order
  actionMap: Map<string, number>; // Map action key to index in actions array
}

export interface UseTriggerEventsReturn {
  // Streaming-only entries (no historical data)
  streamingEntries: TriggerHistoryEntry[];
  isStreamActive: boolean;
  contextInfo: ContextInfo | null;
  setContextInfo: (context: ContextInfo) => void;
  clearStreamingData: () => void;
  updateAction: (
    triggerId: string,
    actionIndex: number,
    updates: Partial<Action>,
  ) => void;
}

function createStreamingAction(
  actionType: Action["type"],
  reasoning: string,
): ActionStreamingData {
  return {
    type: actionType,
    reasoning,
    result: { type: "streaming" as const, result: "" },
    duration_ms: 0,
    start_timestamp: new Date().toISOString(),
  };
}

/**
 * Processes trigger-based streaming events into a timeline of trigger entries.
 * Each trigger entry represents a user input and all the actions taken in response.
 * Only one trigger can be active at a time.
 */
export function useTriggerEvents(
  events: ClientAgentEvent[],
): UseTriggerEventsReturn {
  // Only streaming entries - no historical data
  const [streamingEntries, setStreamingEntries] = useState<
    TriggerHistoryEntry[]
  >([]);
  const [activeTrigger, setActiveTrigger] =
    useState<ActiveTriggerBuilder | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [contextInfo, setContextInfo] = useState<ContextInfo | null>(null);
  const lastProcessedEventId = useRef<number | null>(null);

  useEffect(() => {
    if (events.length === 0) return;

    let numConversationMessages = contextInfo?.conversation_messages || 0;
    let currentTrigger = activeTrigger
      ? {
          ...activeTrigger,
          actions: [...activeTrigger.actions],
          actionMap: new Map(activeTrigger.actionMap),
        }
      : null;
    let hasActiveStreaming = isStreamActive;

    for (const event of events) {
      if (
        lastProcessedEventId.current !== null &&
        event.id <= lastProcessedEventId.current
      ) {
        continue; // Skip already processed events
      }

      lastProcessedEventId.current = event.id;

      debug.log("Processing trigger event:", event);

      switch (event.type) {
        case "trigger_started": {
          // Start a new trigger (should be only one active)
          if (currentTrigger && currentTrigger.entry_id !== event.entry_id) {
            debug.warn(
              "Starting new trigger while another is active. This shouldn't happen.",
            );
          }

          numConversationMessages += 1;
          currentTrigger = {
            entry_id: event.entry_id,
            trigger: event.trigger,
            actions: [],
            actionMap: new Map(),
          };

          hasActiveStreaming = true;
          break;
        }

        case "action_started": {
          // Start tracking a new action
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(
              `Received action_started for unknown entry_id: ${event.entry_id}`,
            );
            continue;
          }

          const actionKey = `${event.sequence_number}_${event.action_number}`;
          const actionIndex = currentTrigger.actions.length;

          currentTrigger.actions.push({
            sequence_number: event.sequence_number,
            action_number: event.action_number,
            partial_results: [],
            action: createStreamingAction(
              event.action_type as Action["type"],
              event.reasoning,
            ),
          });

          currentTrigger.actionMap.set(actionKey, actionIndex);

          hasActiveStreaming = true;
          break;
        }

        case "action_progress": {
          // Update the most recent action of this type with streaming progress
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(
              `Received action_progress for unknown entry_id: ${event.entry_id}`,
            );
            continue;
          }

          // Find the exact action by sequence and action number
          const actionKey = `${event.sequence_number}_${event.action_number}`;
          const actionIndex = currentTrigger.actionMap.get(actionKey);

          if (actionIndex !== undefined) {
            const targetAction = currentTrigger.actions[actionIndex];
            targetAction.partial_results.push(event.partial_result);
            targetAction.action.result = {
              type: "streaming",
              result: targetAction.partial_results.join(""),
            };
          } else {
            debug.warn(
              `Received action_progress for unknown action: ${actionKey} in entry ${event.entry_id}`,
            );
          }

          hasActiveStreaming = true;
          break;
        }

        case "action_completed": {
          // Complete the most recent action of this type
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(
              `Received action_completed for unknown entry_id: ${event.entry_id}`,
            );
            continue;
          }

          // Find the exact action by sequence and action number
          const actionKey = `${event.sequence_number}_${event.action_number}`;
          const actionIndex = currentTrigger.actionMap.get(actionKey);

          if (actionIndex !== undefined) {
            currentTrigger.actions[actionIndex].action = event.action;
          } else {
            debug.warn(
              `Received action_completed for unknown action: ${actionKey} in entry ${event.entry_id}`,
            );
          }

          hasActiveStreaming = true;
          break;
        }

        case "trigger_completed": {
          // Use the complete entry from the event
          const triggerEntry: TriggerHistoryEntry = event.entry;

          if (
            !currentTrigger ||
            currentTrigger.entry_id !== triggerEntry.entry_id
          ) {
            debug.warn(
              `Received trigger_completed for unknown entry_id: ${triggerEntry.entry_id}`,
            );
            continue;
          }

          // Extract and update context info from the trigger completed event
          const newContextInfo: ContextInfo = {
            estimated_tokens: event.estimated_tokens,
            context_limit: event.context_limit,
            usage_percentage: event.usage_percentage,
            approaching_limit: event.approaching_limit,
            conversation_messages: numConversationMessages,
          };
          setContextInfo(newContextInfo);

          // Add to streaming entries and clear active trigger
          setStreamingEntries((prev) => [...prev, triggerEntry]);
          currentTrigger = null;

          hasActiveStreaming = false; // This trigger is complete
          break;
        }

        case "summarization_started":
        case "summarization_finished":
          // Ignore summarization events for now
          break;

        default:
          // Ignore other event types for trigger processing
          break;
      }
    }

    debug.log("Setting activeTrigger and isStreamActive", {
      currentTriggerEntryId: currentTrigger?.entry_id,
      hasActiveStreaming,
      actionsCount: currentTrigger?.actions.length || 0,
    });

    setActiveTrigger(currentTrigger);
    setIsStreamActive(hasActiveStreaming);
  }, [events]);

  // Combine completed streaming entries with active trigger
  const allStreamingEntries = useMemo(() => {
    const entries = [...streamingEntries];

    if (activeTrigger) {
      // Convert active trigger to TriggerHistoryEntry
      const activeActions: Action[] = activeTrigger.actions.map(
        (builder) => builder.action,
      );

      const activeTriggerEntry: TriggerHistoryEntry = {
        trigger: activeTrigger.trigger,
        actions_taken: activeActions,
        timestamp: activeTrigger.trigger.timestamp,
        entry_id: activeTrigger.entry_id,
        situational_context: "", // Not available yet for active triggers
        compressed_summary: null,
        end_timestamp: null,
      };

      entries.push(activeTriggerEntry);
    }

    return entries;
  }, [streamingEntries, activeTrigger]);

  const clearStreamingData = useCallback(() => {
    setStreamingEntries([]);
    setActiveTrigger(null);
    setIsStreamActive(false);
    setContextInfo(null);
    lastProcessedEventId.current = null;
  }, []);

  const updateAction = useCallback(
    (triggerId: string, actionIndex: number, updates: Partial<Action>) => {
      setStreamingEntries((prev) =>
        prev.map((entry) =>
          entry.entry_id === triggerId
            ? {
                ...entry,
                actions_taken: entry.actions_taken.map((action, index) =>
                  index === actionIndex
                    ? ({ ...action, ...updates } as Action)
                    : action,
                ),
              }
            : entry,
        ),
      );
    },
    [],
  );

  return {
    streamingEntries: allStreamingEntries,
    isStreamActive,
    contextInfo,
    setContextInfo,
    clearStreamingData,
    updateAction,
  };
}
