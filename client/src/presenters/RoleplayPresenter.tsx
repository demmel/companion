import { useRef, useMemo } from 'react';
import { ConversationPresenterProps } from './types';
import { AgentMessage, ToolCall } from '../types';
import { RoleplayText } from '../components/RoleplayText';
import { ToolDisplay } from '../components/ToolDisplay';
import { RoleplayState, CharacterState } from '../types/roleplay';

const HIDDEN_TOOLS = new Set(['remember_detail', 'correct_detail']);

const MOOD_EMOJIS: Record<string, string> = {
  happy: '😊', excited: '🤩', playful: '😈', flirtatious: '😘',
  sad: '😢', angry: '😠', frustrated: '😤', annoyed: '🙄',
  nervous: '😰', shy: '😊', confident: '😎', mysterious: '😏',
  seductive: '😍', mischievous: '😋', gentle: '🥰', fierce: '🔥',
  neutral: '😐', curious: '🤔', surprised: '😯', worried: '😟'
};

const MOOD_COLORS: Record<string, string> = {
  happy: 'text-yellow-500', excited: 'text-purple-500', playful: 'text-cyan-500',
  sad: 'text-blue-500', angry: 'text-red-500', neutral: 'text-gray-500'
};

function createInitialRoleplayState(): RoleplayState {
  return {
    current_character_id: null,
    characters: {},
    global_scene: null,
    global_memories: []
  };
}

interface MessageWithState {
  message: AgentMessage | UserMessage;
  index: number;
  stateAtMessage: RoleplayState;
  shouldShowHeader: boolean;
  currentCharacter: CharacterState | null;
  visibleToolCalls: ToolCall[];
}

function buildMessagesWithState(
  messages: (AgentMessage | UserMessage)[], 
  initialState?: RoleplayState
): MessageWithState[] {
  let currentState = initialState || createInitialRoleplayState();
  let lastSpeakingCharacter: string | null = null;
  
  return messages.map((message, index) => {
    // User messages reset the speaking character (user "speaks")
    if (message.role === 'user') {
      lastSpeakingCharacter = null;
    }
    
    // For agent messages, apply tool calls to evolve state
    if (message.role === 'assistant') {
      const agentMessage = message as AgentMessage;
      for (const toolCall of agentMessage.tool_calls) {
        currentState = applyToolCallToState(currentState, toolCall);
      }
    }
    
    const currentCharacterId = currentState.current_character_id;
    const currentCharacter = currentCharacterId ? currentState.characters[currentCharacterId] : null;
    
    // Calculate header visibility for agent messages
    let shouldShowHeader = false;
    let visibleToolCalls: ToolCall[] = [];
    
    if (message.role === 'assistant') {
      const agentMessage = message as AgentMessage;
      visibleToolCalls = agentMessage.tool_calls.filter(tc => !HIDDEN_TOOLS.has(tc.tool_name));
      const hasVisibleContent = !!(agentMessage.content.trim() || 
        visibleToolCalls.some(tc => tc.tool_name !== 'assume_character'));
      
      // Show header when character changes (including from null to character)
      // or when user has spoken since last character message
      shouldShowHeader = !!currentCharacterId && 
        currentCharacterId !== lastSpeakingCharacter &&
        hasVisibleContent;
      
      if (shouldShowHeader) {
        lastSpeakingCharacter = currentCharacterId;
      }
    }
    
    return {
      message,
      index,
      stateAtMessage: currentState,
      shouldShowHeader,
      currentCharacter,
      visibleToolCalls
    };
  });
}

function applyToolCallToState(state: RoleplayState, toolCall: ToolCall): RoleplayState {
  if (toolCall.type !== 'finished') return state;
  
  const newState = { ...state };
  
  switch (toolCall.tool_name) {
    case 'assume_character': {
      const charId = `char_${toolCall.parameters.character_name}`;
      newState.current_character_id = charId;
      newState.characters = {
        ...newState.characters,
        [charId]: {
          id: charId,
          name: toolCall.parameters.character_name,
          personality: toolCall.parameters.personality || '',
          background: toolCall.parameters.background,
          quirks: toolCall.parameters.quirks,
          mood: 'neutral',
          mood_intensity: 'moderate',
          memories: [],
          actions: [],
          thoughts: []
        }
      };
      break;
    }
    
    case 'set_mood': {
      if (newState.current_character_id && newState.characters[newState.current_character_id]) {
        const char = newState.characters[newState.current_character_id];
        newState.characters = {
          ...newState.characters,
          [newState.current_character_id]: {
            ...char,
            mood: toolCall.parameters.mood,
            mood_intensity: toolCall.parameters.intensity || char.mood_intensity
          }
        };
      }
      break;
    }
    
    case 'scene_setting': {
      newState.global_scene = {
        location: toolCall.parameters.location,
        atmosphere: toolCall.parameters.atmosphere,
        time: toolCall.parameters.time
      };
      break;
    }
  }
  
  return newState;
}

export function RoleplayPresenter({ messages, isStreamActive, agentState }: ConversationPresenterProps) {
  // Build chronological state history with header logic
  const messagesWithState = useMemo(() => {
    return buildMessagesWithState(messages, agentState as RoleplayState);
  }, [messages, agentState]);

  return (
    <div className="space-y-2">
      {messagesWithState.map((messageWithState) => {
        const { message, index, shouldShowHeader, currentCharacter, visibleToolCalls, stateAtMessage } = messageWithState;
        
        if (message.role === 'user') {
          return (
            <div key={index} className="mb-4">
              <div className="text-sm text-gray-600 mb-1">You</div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          );
        }

        const agentMessage = message as AgentMessage;
        

        return (
          <AgentMessageItem
            key={index}
            message={agentMessage}
            shouldShowHeader={shouldShowHeader}
            currentCharacter={currentCharacter}
            stateAtMessage={stateAtMessage}
            visibleToolCalls={visibleToolCalls}
          />
        );
      })}
      
      {isStreamActive && (
        <div className="text-gray-500">
          <span className="animate-pulse">▋</span>
        </div>
      )}
    </div>
  );
}

function AgentMessageItem({ 
  message, 
  shouldShowHeader, 
  currentCharacter,
  stateAtMessage,
  visibleToolCalls
}: { 
  message: AgentMessage;
  shouldShowHeader: boolean;
  currentCharacter: CharacterState | null;
  stateAtMessage: RoleplayState;
  visibleToolCalls: ToolCall[];
}) {
  

  return (
    <div className="mb-4">
      {/* Character Header - only when character changes */}
      {shouldShowHeader && currentCharacter && (
        <CharacterHeader character={currentCharacter} stateAtMessage={stateAtMessage} />
      )}
      
      {!currentCharacter && (
        <div className="text-sm text-gray-600 mb-1">🤖 Agent</div>
      )}
      
      {/* Message content */}
      {message.content && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-2">
          <RoleplayText content={message.content} />
        </div>
      )}
      
      {/* Tool presentations */}
      {visibleToolCalls.map((toolCall, index) => (
        <SpecialToolPresentation 
          key={`${toolCall.tool_id}-${index}`} 
          toolCall={toolCall} 
          stateAtMessage={stateAtMessage}
        />
      ))}
    </div>
  );
}

function CharacterHeader({ character, stateAtMessage }: { character: CharacterState; stateAtMessage: RoleplayState }) {
  const moodEmoji = MOOD_EMOJIS[character.mood] || '😐';
  const moodColor = MOOD_COLORS[character.mood] || 'text-gray-500';
  
  return (
    <div className="mb-2">
      <div className={`text-sm font-medium ${moodColor} flex items-center gap-2`}>
        <span>🎭</span>
        <span className="font-bold">{character.name}</span>
        <span className="text-lg">{moodEmoji}</span>
        <span className="text-xs text-gray-500 italic">({character.mood} - {character.mood_intensity})</span>
      </div>
      
      {/* Scene context */}
      {stateAtMessage.global_scene && (
        <div className="text-xs text-gray-500 italic">
          📍 {stateAtMessage.global_scene.location}
          {stateAtMessage.global_scene.atmosphere && ` - ${stateAtMessage.global_scene.atmosphere}`}
        </div>
      )}
    </div>
  );
}

function SpecialToolPresentation({ toolCall, stateAtMessage }: { toolCall: ToolCall; stateAtMessage: RoleplayState }) {
  // Don't show assume_character - handled by header
  if (toolCall.tool_name === 'assume_character') {
    return null;
  }
  
  // Special presentations for key roleplay tools
  switch (toolCall.tool_name) {
    case 'set_mood':
      return <MoodTransition toolCall={toolCall} stateAtMessage={stateAtMessage} />;
    case 'character_action':
      return <CharacterAction toolCall={toolCall} />;
    case 'internal_thought':
      return <InternalThought toolCall={toolCall} />;
    case 'scene_setting':
      return <SceneSetting toolCall={toolCall} />;
    default:
      // Use generic tool display for everything else
      return <ToolDisplay toolCall={toolCall} />;
  }
}

function CharacterAction({ toolCall }: { toolCall: ToolCall }) {
  const action = toolCall.parameters.action;
  if (!action) return null;
  
  return (
    <div className="text-blue-500 italic text-sm py-1">
      *{action}*
    </div>
  );
}

function MoodTransition({ toolCall, stateAtMessage }: { toolCall: ToolCall; stateAtMessage: RoleplayState }) {
  if (toolCall.type !== 'finished') return null;
  
  const newMood = toolCall.parameters.mood || 'neutral';
  
  // Get the previous mood from the current character state before this tool executed
  const currentCharacter = stateAtMessage.current_character_id 
    ? stateAtMessage.characters[stateAtMessage.current_character_id] 
    : null;
  const oldMood = currentCharacter?.mood || 'neutral';
  
  const oldEmoji = MOOD_EMOJIS[oldMood] || '😐';
  const newEmoji = MOOD_EMOJIS[newMood] || '😐';
  const moodColor = MOOD_COLORS[newMood] || 'text-gray-500';
  
  return (
    <div className={`text-sm ${moodColor} py-1`}>
      {oldEmoji} → {newEmoji}
      {toolCall.parameters.flavor_text && (
        <span className="ml-2 opacity-75">{toolCall.parameters.flavor_text}</span>
      )}
    </div>
  );
}

function InternalThought({ toolCall }: { toolCall: ToolCall }) {
  const thought = toolCall.parameters.thought;
  if (!thought) return null;
  
  return (
    <div className="text-yellow-600 opacity-75 text-sm py-1">
      💭 {thought}
    </div>
  );
}

function SceneSetting({ toolCall }: { toolCall: ToolCall }) {
  const { location, atmosphere, time } = toolCall.parameters;
  
  const parts = [];
  if (location) parts.push(location);
  if (atmosphere) parts.push(atmosphere);
  if (time) parts.push(`(${time})`);
  
  if (parts.length === 0) return null;
  
  return (
    <div className="text-purple-600 text-sm py-1 italic">
      📍 {parts.join(' - ')}
    </div>
  );
}