import { Trigger, Action, TimelineEntry, PaginationInfo } from "./types";

export interface AgentErrorEvent {
  message: string;
  tool_name?: string;
  tool_id?: string;
  type: "error";
}

export interface SummarizationStartedEvent {
  messages_to_summarize: number;
  recent_messages_kept: number;
  context_usage_before: number;
  type: "summarization_started";
}

export interface SummarizationFinishedEvent {
  summary: string;
  messages_summarized: number;
  messages_after: number;
  context_usage_after: number;
  type: "summarization_finished";
}

// New trigger-based streaming events
export interface TriggerStartedEvent {
  trigger: Trigger;
  entry_id: string;
  timestamp: string;
  type: "trigger_started";
}

export interface TriggerCompletedEvent {
  entry_id: string;
  total_actions: number;
  successful_actions: number;
  timestamp: string;
  type: "trigger_completed";
  // Context information for UI updates
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  approaching_limit: boolean;
}

export interface ActionStartedEvent {
  entry_id: string;
  action_type: string;
  context_given: string;
  sequence_number: number;
  action_number: number;
  timestamp: string;
  reasoning: string;
  type: "action_started";
}

export interface ActionProgressEvent {
  entry_id: string;
  action_type: string;
  partial_result: string;
  sequence_number: number;
  action_number: number;
  timestamp: string;
  type: "action_progress";
}

export interface ActionCompletedEvent {
  entry_id: string;
  action: Action;
  sequence_number: number;
  action_number: number;
  timestamp: string;
  type: "action_completed";
}

// Union type for all agent events (NOT including server event wrappers)
export type AgentEvent =
  | AgentErrorEvent
  | SummarizationStartedEvent
  | SummarizationFinishedEvent
  | TriggerStartedEvent
  | TriggerCompletedEvent
  | ActionStartedEvent
  | ActionProgressEvent
  | ActionCompletedEvent;

// Server event wrappers for hydration protocol
export interface EventEnvelope {
  type: "event_envelope";
  event_sequence: number;
  trigger_id: string;
  event: AgentEvent;
}

export interface HydrationResponse {
  type: "hydration_response";
  entries: TimelineEntry[];
  pagination: PaginationInfo;
}

// Discriminated union for all events sent from server
export type AgentServerEvent = HydrationResponse | EventEnvelope;
