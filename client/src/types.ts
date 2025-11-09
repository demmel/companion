// Trigger-based types
export interface UserInputTrigger {
  type: "user_input";
  content: string;
  user_name: string;
  timestamp: string;
  image_urls?: string[];
}

export interface WakeupTrigger {
  type: "wakeup";
  timestamp: string;
}

export type Trigger = UserInputTrigger | WakeupTrigger;

export type ActionStatus =
  | { type: "streaming"; result: string }
  | { type: "success"; result: string }
  | { type: "error"; error: string };

export interface BaseAction {
  context_given: string;
  reasoning: string;
  status: ActionStatus;
  duration_ms: number;
}

export interface ThinkAction extends BaseAction {
  type: "think";
}

export interface SpeakAction extends BaseAction {
  type: "speak";
}

export interface UpdateAppearanceAction extends BaseAction {
  type: "update_appearance";
  image_description: string;
  image_url: string;
}

export interface UpdateEnvironmentAction extends BaseAction {
  type: "update_environment";
  image_description: string;
  image_url: string;
}

export interface UpdateMoodAction extends BaseAction {
  type: "update_mood";
}

export interface WaitAction extends BaseAction {
  type: "wait";
}

export interface CreativeInspirationAction extends BaseAction {
  type: "get_creative_inspiration";
  words: string[];
}

export interface AddPriorityAction extends BaseAction {
  type: "add_priority";
}

export interface RemovePriorityAction extends BaseAction {
  type: "remove_priority";
}

export interface PriorityOperationResult {
  operation_type: "add" | "remove" | "merge" | "refine" | "reorder";
  summary: string;
}

export interface EvaluatePrioritiesAction extends BaseAction {
  type: "evaluate_priorities";
  operations: PriorityOperationResult[];
}

export interface FetchUrlAction extends BaseAction {
  type: "fetch_url";
  url: string;
  looking_for: string;
}

export interface SearchResult {
  url: string;
  title: string;
  snippet: string;
}

export interface SearchWebAction extends BaseAction {
  type: "search_web";
  query: string;
  purpose: string;
  search_results: SearchResult[];
}

export type Action =
  | ThinkAction
  | SpeakAction
  | UpdateAppearanceAction
  | UpdateEnvironmentAction
  | UpdateMoodAction
  | WaitAction
  | CreativeInspirationAction
  | AddPriorityAction
  | RemovePriorityAction
  | EvaluatePrioritiesAction
  | FetchUrlAction
  | SearchWebAction;

export interface Summary {
  summary_text: string;
  insert_at_index: number;
  created_at: string;
  status: "in_progress" | "completed";
  messages_to_summarize: number;
  recent_messages_kept: number;
}

export interface TriggerHistoryEntry {
  trigger: Trigger;
  actions_taken: Action[];
  timestamp: string;
  entry_id: string;
  situational_context: string;
  compressed_summary?: string;
}

export interface TriggerHistoryResponse {
  entries: TriggerHistoryEntry[];
  summaries: Summary[];
  total_entries: number;
  recent_entries_count: number;
}

export interface ContextInfo {
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  conversation_messages: number;
  approaching_limit: boolean;
}

// Timeline pagination types
export interface TimelineEntryTrigger {
  type: "trigger";
  entry: TriggerHistoryEntry;
}

export interface TimelineEntrySummary {
  type: "summary";
  summary: Summary;
}

export type TimelineEntry = TimelineEntryTrigger;

export interface PaginationInfo {
  total_items: number;
  page_size: number;
  has_next: boolean;
  has_previous: boolean;
  next_cursor?: string;
  previous_cursor?: string;
}

export interface TimelineResponse {
  entries: TimelineEntry[];
  pagination: PaginationInfo;
}

// Auto-wakeup types
export interface AutoWakeupStatusResponse {
  enabled: boolean;
  delay_seconds: number;
}

export interface AutoWakeupSetRequest {
  enabled: boolean;
}

export interface AutoWakeupSetResponse {
  enabled: boolean;
  message: string;
  timestamp: string;
}

// Image upload types
export interface ImageUploadResponse {
  id: string;
  size: number;
  url: string;
}

// Model configuration types
export interface ModelConfigResponse {
  state_initialization_model: string;
  action_planning_model: string;
  situational_analysis_model: string;
  memory_retrieval_model: string;
  memory_formation_model: string;
  trigger_compression_model: string;
  think_action_model: string;
  speak_action_model: string;
  visual_action_model: string;
  fetch_url_action_model: string;
  evaluate_priorities_action_model: string;
}

export interface ModelConfigUpdateRequest {
  state_initialization_model: string;
  action_planning_model: string;
  situational_analysis_model: string;
  memory_retrieval_model: string;
  memory_formation_model: string;
  trigger_compression_model: string;
  think_action_model: string;
  speak_action_model: string;
  visual_action_model: string;
  fetch_url_action_model: string;
  evaluate_priorities_action_model: string;
}

export interface ModelConfigUpdateResponse {
  message: string;
  timestamp: string;
}

export interface SupportedModelsResponse {
  models: string[];
}
