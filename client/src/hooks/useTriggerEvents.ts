import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import {
  TriggerHistoryEntry,
  Action,
  UpdateAppearanceAction,
  UpdateEnvironmentAction,
  FetchUrlAction,
  SearchWebAction,
  Trigger,
  ContextInfo,
  BaseAction,
} from "../types";
import { debug } from "@/utils/debug";

// Helper to build action objects from streaming events
interface BaseActionBuilder extends BaseAction {
  sequence_number: number;
  action_number: number;
  partial_results: string[];
}

type PartialAction = Partial<Action>;
type ActionBuilder = PartialAction & {
  type: Action["type"];
} & BaseActionBuilder;

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
}

/**
 * Converts an ActionBuilder to a proper Action object for display
 */
function convertActionBuilderToAction(actionBuilder: ActionBuilder): Action {
  const baseAction = {
    ...actionBuilder,
  };

  switch (actionBuilder.type) {
    case "think":
    case "speak":
    case "update_mood":
    case "wait":
    case "get_creative_inspiration":
    case "add_priority":
    case "remove_priority":
      return {
        ...baseAction,
      } as Action;
    case "update_appearance":
      return {
        ...baseAction,
        image_description: actionBuilder.image_description,
        image_url: actionBuilder.image_url,
      } as UpdateAppearanceAction;
    case "update_environment":
      return {
        ...baseAction,
        image_description: actionBuilder.image_description,
        image_url: actionBuilder.image_url,
      } as UpdateEnvironmentAction;
    case "fetch_url":
      return {
        ...baseAction,
        url: actionBuilder.url || "",
        looking_for: actionBuilder.looking_for || "",
      } as FetchUrlAction;
    case "search_web":
      return {
        ...baseAction,
        query: actionBuilder.query || "",
        purpose: actionBuilder.purpose || "",
        search_results: actionBuilder.search_results || [],
      } as SearchWebAction;
    default:
      const exhaustiveCheck: never = actionBuilder;
      throw new Error(
        `Unknown action type: ${(exhaustiveCheck as ActionBuilder).type}`,
      );
  }
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
            status: {
              type: "streaming",
              result: "",
            },
            type: event.action_type as ActionBuilder["type"],
            context_given: event.context_given,
            duration_ms: 0, // Duration will be updated later
            partial_results: [],
            reasoning: event.reasoning,
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
            targetAction.status = {
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
            const targetAction = {
              ...currentTrigger.actions[actionIndex],
              ...event.action,
            };

            currentTrigger.actions[actionIndex] = targetAction;
          } else {
            debug.warn(
              `Received action_completed for unknown action: ${actionKey} in entry ${event.entry_id}`,
            );
          }

          hasActiveStreaming = true;
          break;
        }

        case "trigger_completed": {
          // Complete the trigger and convert to final entry
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(
              `Received trigger_completed for unknown entry_id: ${event.entry_id}`,
            );
            continue;
          }

          if (!currentTrigger.trigger) {
            debug.warn(
              `Trigger completed but no trigger data found for entry_id: ${event.entry_id}`,
            );
            continue;
          }

          // Convert action builders to final actions (sorted by sequence, then action number)
          const actions: Action[] = [];
          const sortedActions = [...currentTrigger.actions].sort((a, b) => {
            if (a.sequence_number !== b.sequence_number) {
              return a.sequence_number - b.sequence_number;
            }
            return a.action_number - b.action_number;
          });

          for (const actionBuilder of sortedActions) {
            if (actionBuilder.status.type !== "streaming") {
              actions.push(convertActionBuilderToAction(actionBuilder));
            }
          }

          // Create final trigger entry
          const triggerEntry: TriggerHistoryEntry = {
            trigger: currentTrigger.trigger,
            actions_taken: actions,
            timestamp: currentTrigger.trigger.timestamp,
            entry_id: event.entry_id,
          };

          // Extract and update context info from the trigger completed event
          const newContextInfo: ContextInfo = {
            estimated_tokens: event.estimated_tokens,
            context_limit: event.context_limit,
            usage_percentage: event.usage_percentage,
            approaching_limit: event.approaching_limit,
            conversation_messages: event.total_actions, // Use total_actions as proxy
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

    console.log(
      `[${new Date().toISOString()}] Setting activeTrigger and isStreamActive`,
      {
        currentTriggerEntryId: currentTrigger?.entry_id,
        hasActiveStreaming,
        actionsCount: currentTrigger?.actions.length || 0,
      },
    );

    setActiveTrigger(currentTrigger);
    setIsStreamActive(hasActiveStreaming);
  }, [events]);

  // Combine completed streaming entries with active trigger
  const allStreamingEntries = useMemo(() => {
    const entries = [...streamingEntries];

    if (activeTrigger) {
      // Convert active trigger to TriggerHistoryEntry
      const activeActions: Action[] = activeTrigger.actions.map(
        convertActionBuilderToAction,
      );

      const activeTriggerEntry: TriggerHistoryEntry = {
        trigger: activeTrigger.trigger,
        actions_taken: activeActions,
        timestamp: activeTrigger.trigger.timestamp,
        entry_id: activeTrigger.entry_id,
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

  return {
    streamingEntries: allStreamingEntries,
    isStreamActive,
    contextInfo,
    setContextInfo,
    clearStreamingData,
  };
}
