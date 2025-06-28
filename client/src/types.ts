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

export type Message = UserMessage | AgentMessage;

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
  | 'response_complete';

export interface BaseAgentEvent {
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

export interface ResponseCompleteEvent extends BaseAgentEvent {
  type: 'response_complete';
}

export type AgentEvent = 
  | AgentTextEvent 
  | ToolStartedEvent 
  | ToolFinishedEvent 
  | AgentErrorEvent 
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