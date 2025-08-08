import { useState, useEffect, useRef, useCallback } from "react";
import { ClientAgentEvent } from "./useWebSocket";
import {
  TriggerHistoryEntry,
  Action,
  ThinkAction,
  SpeakAction,
  UpdateAppearanceAction,
  UpdateMoodAction,
  WaitAction,
  Trigger,
  ActionStatus,
} from "../types";
import { debug } from "@/utils/debug";

// Helper to build action objects from streaming events
interface BaseActionBuilder {
  sequence_number: number;
  action_number: number;
  context_given: string;
  status: ActionStatus;
  duration_ms: number;
  partial_results: string[];
}

interface ThinkActionBuilder extends BaseActionBuilder {
  action_type: "think";
}

interface SpeakActionBuilder extends BaseActionBuilder {
  action_type: "speak";
}

interface UpdateAppearanceActionBuilder extends BaseActionBuilder {
  action_type: "update_appearance";
  image_description?: string;
  image_url?: string;
}

interface UpdateMoodActionBuilder extends BaseActionBuilder {
  action_type: "update_mood";
}

interface WaitActionBuilder extends BaseActionBuilder {
  action_type: "wait";
}

type ActionBuilder =
  | ThinkActionBuilder
  | SpeakActionBuilder
  | UpdateAppearanceActionBuilder
  | UpdateMoodActionBuilder
  | WaitActionBuilder;

// Single active trigger builder (only one trigger can be active at a time)
interface ActiveTriggerBuilder {
  entry_id: string;
  trigger: Trigger;
  actions: ActionBuilder[]; // Array to maintain execution order
  actionMap: Map<string, number>; // Map action key to index in actions array
}

export interface UseTriggerEventsReturn {
  triggerEntries: TriggerHistoryEntry[];
  isStreamActive: boolean;
  loadTriggerHistory: (entries: TriggerHistoryEntry[]) => void;
  clearTriggerHistory: () => void;
}

/**
 * Converts an ActionBuilder to a proper Action object for display
 */
function convertActionBuilderToAction(actionBuilder: ActionBuilder): Action {
  const baseAction = {
    context_given: actionBuilder.context_given,
    status: actionBuilder.status,
    duration_ms: actionBuilder.duration_ms || 0,
  };

  switch (actionBuilder.action_type) {
    case "think":
      return {
        type: "think",
        ...baseAction,
      } as ThinkAction;

    case "speak":
      return {
        type: "speak",
        ...baseAction,
      } as SpeakAction;

    case "update_appearance":
      return {
        type: "update_appearance",
        ...baseAction,
        image_description: actionBuilder.image_description,
        image_url: actionBuilder.image_url,
      } as UpdateAppearanceAction;

    case "update_mood":
      return {
        type: "update_mood",
        ...baseAction,
      } as UpdateMoodAction;

    case "wait":
      return {
        type: "wait",
        ...baseAction,
      } as WaitAction;

    default:
      throw new Error(`Unknown action type: ${(actionBuilder as ActionBuilder).action_type}`);
  }
}

/**
 * Processes trigger-based streaming events into a timeline of trigger entries.
 * Each trigger entry represents a user input and all the actions taken in response.
 * Only one trigger can be active at a time.
 */
export function useTriggerEvents(events: ClientAgentEvent[]): UseTriggerEventsReturn {
  const [triggerEntries, setTriggerEntries] = useState<TriggerHistoryEntry[]>([]);
  const [activeTrigger, setActiveTrigger] = useState<ActiveTriggerBuilder | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const lastProcessedEventId = useRef<number | null>(null);

  useEffect(() => {
    if (events.length === 0) return;

    let currentTrigger = activeTrigger;
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
            debug.warn("Starting new trigger while another is active. This shouldn't happen.");
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
            debug.warn(`Received action_started for unknown entry_id: ${event.entry_id}`);
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
            action_type: event.action_type as "think" | "speak" | "update_appearance" | "update_mood" | "wait",
            context_given: event.context_given,
            duration_ms: 0, // Duration will be updated later
            partial_results: [],
          });

          currentTrigger.actionMap.set(actionKey, actionIndex);

          hasActiveStreaming = true;
          break;
        }

        case "action_progress": {
          // Update the most recent action of this type with streaming progress
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(`Received action_progress for unknown entry_id: ${event.entry_id}`);
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
            debug.warn(`Received action_progress for unknown action: ${actionKey} in entry ${event.entry_id}`);
          }

          hasActiveStreaming = true;
          break;
        }

        case "action_completed": {
          // Complete the most recent action of this type
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(`Received action_completed for unknown entry_id: ${event.entry_id}`);
            continue;
          }

          // Find the exact action by sequence and action number
          const actionKey = `${event.sequence_number}_${event.action_number}`;
          const actionIndex = currentTrigger.actionMap.get(actionKey);

          if (actionIndex !== undefined) {
            const targetAction = currentTrigger.actions[actionIndex];

            // Extract data from ActionDTO and populate the builder
            const action = event.action;
            targetAction.status = action.status;
            targetAction.duration_ms = action.duration_ms;

            // Populate type-specific fields
            if (action.type === "update_appearance" && targetAction.action_type === "update_appearance") {
              const appearanceBuilder = targetAction as UpdateAppearanceActionBuilder;
              appearanceBuilder.image_description = action.image_description;
              appearanceBuilder.image_url = action.image_url;
            }
          } else {
            debug.warn(`Received action_completed for unknown action: ${actionKey} in entry ${event.entry_id}`);
          }

          hasActiveStreaming = true;
          break;
        }

        case "trigger_completed": {
          // Complete the trigger and convert to final entry
          if (!currentTrigger || currentTrigger.entry_id !== event.entry_id) {
            debug.warn(`Received trigger_completed for unknown entry_id: ${event.entry_id}`);
            continue;
          }

          if (!currentTrigger.trigger) {
            debug.warn(`Trigger completed but no trigger data found for entry_id: ${event.entry_id}`);
            continue;
          }

          // Convert action builders to final actions (sorted by sequence, then action number)
          const actions: Action[] = [];
          const sortedActions = [...currentTrigger.actions].sort(
            (a, b) => {
              if (a.sequence_number !== b.sequence_number) {
                return a.sequence_number - b.sequence_number;
              }
              return a.action_number - b.action_number;
            }
          );

          for (const actionBuilder of sortedActions) {
            if (actionBuilder.status.type !== "streaming") {
              const baseAction = {
                context_given: actionBuilder.context_given,
                status: actionBuilder.status,
                duration_ms: actionBuilder.duration_ms || 0,
              };

              switch (actionBuilder.action_type) {
                case "think":
                  actions.push({
                    type: "think",
                    ...baseAction,
                  } as ThinkAction);
                  break;

                case "speak":
                  actions.push({
                    type: "speak",
                    ...baseAction,
                  } as SpeakAction);
                  break;

                case "update_appearance":
                  actions.push({
                    type: "update_appearance",
                    ...baseAction,
                    image_description: actionBuilder.image_description,
                    image_url: actionBuilder.image_url,
                  } as UpdateAppearanceAction);
                  break;

                case "update_mood":
                  actions.push({
                    type: "update_mood",
                    ...baseAction,
                  } as UpdateMoodAction);
                  break;

                case "wait":
                  actions.push({
                    type: "wait",
                    ...baseAction,
                  } as WaitAction);
                  break;

                default:
                  throw new Error(`Unknown action type: ${(actionBuilder as ActionBuilder).action_type}`);
              }
            }
          }

          // Create final trigger entry
          const triggerEntry: TriggerHistoryEntry = {
            trigger: currentTrigger.trigger,
            actions_taken: actions,
            timestamp: currentTrigger.trigger.timestamp,
            entry_id: event.entry_id,
          };

          // Add to completed entries and clear active trigger
          setTriggerEntries(prev => [...prev, triggerEntry]);
          currentTrigger = null;

          hasActiveStreaming = false; // This trigger is complete
          break;
        }

        default:
          // Ignore other event types for trigger processing
          break;
      }
    }

    setActiveTrigger(currentTrigger);
    setIsStreamActive(hasActiveStreaming);
  }, [events, activeTrigger]);

  // Combine completed triggers with active trigger
  const allTriggerEntries = [...triggerEntries];

  if (activeTrigger) {
    // Convert active trigger to TriggerHistoryEntry
    const activeActions: Action[] = activeTrigger.actions.map(convertActionBuilderToAction);

    const activeTriggerEntry: TriggerHistoryEntry = {
      trigger: activeTrigger.trigger,
      actions_taken: activeActions,
      timestamp: activeTrigger.trigger.timestamp,
      entry_id: activeTrigger.entry_id,
    };

    allTriggerEntries.push(activeTriggerEntry);
  }

  const loadTriggerHistory = useCallback((entries: TriggerHistoryEntry[]) => {
    setTriggerEntries(entries);
    setActiveTrigger(null);
    setIsStreamActive(false);
    lastProcessedEventId.current = null;
  }, []);

  const clearTriggerHistory = useCallback(() => {
    setTriggerEntries([]);
    setActiveTrigger(null);
    setIsStreamActive(false);
    lastProcessedEventId.current = null;
  }, []);

  return {
    triggerEntries: allTriggerEntries,
    isStreamActive,
    loadTriggerHistory,
    clearTriggerHistory,
  };
}