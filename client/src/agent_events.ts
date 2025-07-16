import { ToolResult } from "./types";

export interface AgentTextEvent {
  content: string;
  is_thought: boolean;
  type: "text";
}

export interface ToolStartedEvent {
  tool_name: string;
  tool_id: string;
  parameters: Record<string, any>;
  type: "tool_started";
}

export interface ToolProgressEvent {
  tool_id: string;
  data: string;
  progress?: number; // 0.0 to 1.0
  type: "tool_progress";
}

export interface ToolFinishedEvent {
  tool_id: string;
  result: ToolResult;
  type: "tool_finished";
}

export interface UserInputRequestEvent {
  // TODO: Define structure when we implement interactive tools
  type: "user_input_request";
}

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

export interface ResponseCompleteEvent {
  message_count: number;
  conversation_messages: number;
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  approaching_limit: boolean;
  type: "response_complete";
}

// Union type for all agent events
export type AgentEvent =
  | AgentTextEvent
  | ToolStartedEvent
  | ToolProgressEvent
  | ToolFinishedEvent
  | UserInputRequestEvent
  | AgentErrorEvent
  | SummarizationStartedEvent
  | SummarizationFinishedEvent
  | ResponseCompleteEvent;
