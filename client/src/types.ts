// API Types - Structured Message Format (matches server message.py)
export interface ToolCallResult {
  type: 'success' | 'error';
  content: string;
}

export interface ToolCallBase {
  tool_name: string;
  tool_id: string;
  parameters: Record<string, any>;
}

export interface ToolCallRunning extends ToolCallBase {
  type: 'started';
  progress?: any;
}

export interface ToolCallFinished extends ToolCallBase {
  type: 'finished';
  result: ToolCallResult;
}

export type ToolCall = ToolCallRunning | ToolCallFinished;

export interface UserMessage {
  role: 'user';
  content: string;
}

export interface AgentMessage {
  role: 'assistant';
  content: string;
  tool_calls: ToolCall[];
}

export interface SummarizationContent {
  type: 'summarization';
  title: string;
  summary: string;
  messages_summarized: number;
  context_usage_before: number;
  context_usage_after: number;
}

export interface TextContent {
  type: 'text';
  text: string;
}

export type SystemContent = SummarizationContent | TextContent | string; // string for backward compatibility

export interface SystemMessage {
  role: 'system';
  content: SystemContent;
}

export type Message = UserMessage | AgentMessage | SystemMessage;

export interface ConversationResponse {
  messages: Message[];
}

export interface AgentConfig {
  name: string;
  description: string;
  tools: string[];
}

export interface ConfigsResponse {
  configs: { [key: string]: string };
}

// WebSocket Event Types
export type AgentEventType = 
  | 'text'
  | 'tool_started'
  | 'tool_finished'
  | 'error'
  | 'summarization_started'
  | 'summarization_finished'
  | 'response_complete';

export interface BaseAgentEvent {
  id: number;
  type: AgentEventType;
}

export interface AgentTextEvent extends BaseAgentEvent {
  type: 'text';
  content: string;
}

export interface ToolStartedEvent extends BaseAgentEvent {
  type: 'tool_started';
  tool_name: string;
  tool_id: string;
  parameters: { [key: string]: any };
}

export interface ToolFinishedEvent extends BaseAgentEvent {
  type: 'tool_finished';
  tool_id: string;
  result_type: 'success' | 'error';
  result: string;
}

export interface AgentErrorEvent extends BaseAgentEvent {
  type: 'error';
  message: string;
  tool_name?: string;
  tool_id?: string;
}

export interface SummarizationStartedEvent extends BaseAgentEvent {
  type: 'summarization_started';
  messages_to_summarize: number;
  recent_messages_kept: number;
  context_usage_before: number;
}

export interface SummarizationFinishedEvent extends BaseAgentEvent {
  type: 'summarization_finished';
  summary: string;
  messages_summarized: number;
  messages_after: number;
  context_usage_after: number;
}

export interface ResponseCompleteEvent extends BaseAgentEvent {
  type: 'response_complete';
}

export type AgentEvent = 
  | AgentTextEvent 
  | ToolStartedEvent 
  | ToolFinishedEvent 
  | AgentErrorEvent 
  | SummarizationStartedEvent
  | SummarizationFinishedEvent
  | ResponseCompleteEvent;

// UI State Types
export interface ToolExecution {
  id: string;
  name: string;
  parameters: { [key: string]: any };
  status: 'running' | 'completed' | 'error';
  result?: string;
  error?: string;
}