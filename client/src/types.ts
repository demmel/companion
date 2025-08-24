// Trigger-based types
export interface UserInputTrigger {
  type: "user_input";
  content: string;
  user_name: string;
  timestamp: string;
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

interface BaseAction {
  context_given: string;
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

export interface UpdateMoodAction extends BaseAction {
  type: "update_mood";
}

export interface WaitAction extends BaseAction {
  type: "wait";
}

export interface AddPriorityAction extends BaseAction {
  type: "add_priority";
}

export interface RemovePriorityAction extends BaseAction {
  type: "remove_priority";
}

export interface FetchUrlAction extends BaseAction {
  type: "fetch_url";
  url: string;
  looking_for: string;
}

export type Action =
  | ThinkAction
  | SpeakAction
  | UpdateAppearanceAction
  | UpdateMoodAction
  | WaitAction
  | AddPriorityAction
  | RemovePriorityAction
  | FetchUrlAction;

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

export type TimelineEntry = TimelineEntryTrigger | TimelineEntrySummary;

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