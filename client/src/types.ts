export interface TextToolContent {
  type: "text";
  text: string;
}

export interface ImageGenerationToolContent {
  type: "image_generated";
  prompt: string; // Final optimized SDXL prompt used
  image_path: string;
  image_url: string; // URL to access the generated image
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  negative_prompt?: string;
  seed?: number;
  
  // New optimization metadata
  original_description?: string; // Original natural description from agent
  optimization_confidence?: number; // LLM confidence in optimization
  camera_angle?: string; // Camera angle chosen
  viewpoint?: string; // Viewpoint chosen
  optimization_notes?: string; // Notes about optimization choices
}

export type ToolContent = TextToolContent | ImageGenerationToolContent;

export interface ToolCallSuccess {
  type: "success";
  content: ToolContent;
}

export interface ToolCallError {
  type: "error";
  error: string;
}

export type ToolResult = ToolCallSuccess | ToolCallError;

export interface ToolCallBase {
  /** Base class for tool call events */
  tool_name: string;
  tool_id: string;
  parameters: Record<string, any>;
}

export interface ToolCallRunning extends ToolCallBase {
  /** Event when a tool call starts */
  type: "started";
  progress?: any; // Optional progress data
}

export interface ToolCallFinished extends ToolCallBase {
  /** Event when a tool call finishes */
  type: "finished";
  result: ToolResult;
}

export type ToolCall = ToolCallRunning | ToolCallFinished;

export interface TextContent {
  /** Structured content for text messages */
  type: "text";
  text: string;
}

export interface SummarizationContent {
  /** Structured content for summarization system messages */
  type: "summarization";
  title: string;
  summary: string;
}

export type UserContentItem = TextContent;
export type AgentContentItem = TextContent;
export type SystemContentItem = SummarizationContent | TextContent;

export type UserContent = UserContentItem[];
export type AgentContent = AgentContentItem[];
export type SystemContent = SystemContentItem[];

export interface UserMessage {
  /** Message from the user */
  role: "user";
  content: UserContent;
}

export interface AgentMessage {
  /** Message from the agent */
  role: "assistant";
  content: AgentContent;
  tool_calls: ToolCall[];
}

export interface SystemMessage {
  /** System message for summarization or other system-level content */
  role: "system";
  content: SystemContent;
}

export type Message = UserMessage | AgentMessage | SystemMessage;
