// Roleplay-specific tool parameter types matching server schemas

export interface AssumeCharacterParams {
  character_name: string;
  personality: string;
  background?: string;
  quirks?: string;
}

export interface SetMoodParams {
  mood: string;
  intensity?: string;
  flavor_text?: string;
}

export interface RememberDetailParams {
  detail: string;
  category?: string;
}

export interface InternalThoughtParams {
  thought: string;
}

export interface RelationshipStatusParams {
  relationship: string;
  feelings?: string;
}

export interface SceneSettingParams {
  location: string;
  atmosphere?: string;
  time?: string;
}

export interface CharacterActionParams {
  action: string;
  reason?: string;
}

export interface EmotionalReactionParams {
  emotion: string;
  trigger?: string;
  intensity?: string;
}

export interface SwitchCharacterParams {
  character_name: string;
}

export interface CorrectDetailParams {
  old_detail: string;
  new_detail: string;
  category?: string;
}

// Roleplay state structure matching server character_state.py
export interface CharacterState {
  id: string;
  name: string;
  personality: string;
  background?: string;
  quirks?: string;
  mood: string;
  mood_intensity: string;
  memories: Array<{
    detail: string;
    category: string;
    timestamp: string;
  }>;
  actions: Array<{
    action: string;
    reason?: string;
    timestamp: string;
  }>;
  thoughts: Array<{
    thought: string;
    timestamp: string;
  }>;
}

export interface RoleplayState {
  current_character_id: string | null;
  characters: Record<string, CharacterState>;
  global_scene: {
    location?: string;
    atmosphere?: string;
    time?: string;
  } | null;
  global_memories: string[];
}
