import { useMemo } from 'react';
import { ConversationPresenterProps } from './types';
import { AgentMessage, UserMessage, ToolCall } from '@/types';
import { RoleplayText } from '@/components/RoleplayText';
import { RoleplayState, CharacterState } from '@/types/roleplay';
import { css } from '@styled-system/css';
// import { demoMessages } from './demoData'; // Available for testing

const HIDDEN_TOOLS = new Set(['assume_character']);
const SYSTEM_TOOLS = new Set(['scene_setting', 'correct_detail', 'remember_detail']);
const AGENT_TOOLS = new Set(['set_mood', 'character_action', 'internal_thought']);

const MOOD_EMOJIS: Record<string, string> = {
  happy: '😊', excited: '🤩', playful: '😈', flirtatious: '😘',
  joyful: '😊', cheerful: '😊', elated: '🤩', ecstatic: '🤩',
  sad: '😢', angry: '😠', frustrated: '😤', annoyed: '🙄',
  nervous: '😰', shy: '😊', confident: '😎', mysterious: '😏',
  seductive: '😍', mischievous: '😋', gentle: '🥰', fierce: '🔥',
  neutral: '😐', curious: '🤔', surprised: '😯', worried: '😟'
};

// const MOOD_COLORS: Record<string, string> = {
//   happy: 'yellow.500', excited: 'purple.500', playful: 'cyan.500',
//   sad: 'blue.500', angry: 'red.500', neutral: 'gray.500'
// }; // Currently unused

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
  stateBeforeMessage: RoleplayState;
  shouldShowHeader: boolean;
  currentCharacter: CharacterState | null;
  visibleToolCalls: ToolCall[];
}

interface MessageBubble {
  role: 'user' | 'assistant' | 'system';
  messages: MessageWithState[];
  shouldShowHeader: boolean;
  currentCharacter: CharacterState | null;
  stateAtMessage: RoleplayState;
  systemTools?: ToolCall[]; // For system-only bubbles
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
    
    // Store state before applying tool calls for this message
    const stateBeforeMessage = currentState;
    
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
      stateBeforeMessage,
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

function groupMessagesIntoBubbles(messagesWithState: MessageWithState[]): MessageBubble[] {
  const bubbles: MessageBubble[] = [];
  let currentBubble: MessageBubble | null = null;

  for (const messageWithState of messagesWithState) {
    const { message, visibleToolCalls } = messageWithState;
    
    // Separate system tools from agent tools
    const systemTools = visibleToolCalls.filter(tc => SYSTEM_TOOLS.has(tc.tool_name));
    const agentTools = visibleToolCalls.filter(tc => AGENT_TOOLS.has(tc.tool_name));
    
    // Update messageWithState to only include agent tools
    const agentMessageWithState = {
      ...messageWithState,
      visibleToolCalls: agentTools
    };

    // Handle agent/user message (if has content or agent tools)
    if (message.content || agentTools.length > 0) {
      // If this is a different role or we don't have a current bubble, start a new one
      if (!currentBubble || currentBubble.role !== message.role) {
        // Finalize previous bubble
        if (currentBubble) {
          bubbles.push(currentBubble);
        }

        // Start new bubble
        currentBubble = {
          role: message.role,
          messages: [agentMessageWithState],
          shouldShowHeader: messageWithState.shouldShowHeader,
          currentCharacter: messageWithState.currentCharacter,
          stateAtMessage: messageWithState.stateAtMessage
        };
      } else {
        // Add to current bubble
        currentBubble.messages.push(agentMessageWithState);
        
        // Update bubble properties with latest message info
        currentBubble.stateAtMessage = messageWithState.stateAtMessage;
        
        // Show header if any message in the bubble should show it
        if (messageWithState.shouldShowHeader) {
          currentBubble.shouldShowHeader = true;
          currentBubble.currentCharacter = messageWithState.currentCharacter;
        }
      }
    }
    
    // Handle system tools as separate system bubbles
    if (systemTools.length > 0) {
      // Finalize current bubble if it exists
      if (currentBubble) {
        bubbles.push(currentBubble);
        currentBubble = null;
      }
      
      // Create system bubble
      bubbles.push({
        role: 'system',
        messages: [], // System bubbles don't have message content
        shouldShowHeader: false,
        currentCharacter: messageWithState.currentCharacter,
        stateAtMessage: messageWithState.stateAtMessage,
        systemTools
      });
    }
  }

  // Don't forget the last bubble
  if (currentBubble) {
    bubbles.push(currentBubble);
  }

  return bubbles;
}

function UserBubble({ bubble }: { bubble: MessageBubble }) {
  return (
    <div className={css({ mb: 4 })}>
      {/* User header */}
      <div className={css({ 
        mb: 3,
        display: 'flex',
        justifyContent: 'flex-end'
      })}>
        <span className={css({ 
          fontWeight: 'medium', 
          color: 'gray.300',
          fontSize: 'sm'
        })}>You</span>
      </div>
      
      <div className={css({ 
        display: 'flex', 
        justifyContent: 'flex-end' 
      })}>
        <div className={css({ 
          maxWidth: { base: 'xs', lg: 'md' }
        })}>
          <div className={css({ 
            bg: 'blue.600', 
            color: 'white', 
            rounded: '2xl',
            roundedTopRight: 'sm',
            px: 4, 
            py: 2 
          })}>
            {bubble.messages.map((messageWithState, index) => (
              <div key={index} className={css({ 
                whiteSpace: 'pre-wrap', 
                fontSize: 'sm',
                '&:not(:last-child)': { mb: 2 }
              })}>
                {messageWithState.message.content}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function AgentBubble({ bubble }: { bubble: MessageBubble }) {
  return (
    <div className={css({ mb: 4 })}>
      {/* Character Header - only when character changes */}
      {bubble.shouldShowHeader && bubble.currentCharacter && (
        <CharacterHeader character={bubble.currentCharacter} stateAtMessage={bubble.stateAtMessage} />
      )}
      
      {/* Message content */}
      <div>
        <div className={css({ 
          bg: 'gray.800', 
          color: 'gray.100', 
          rounded: '2xl',
          roundedTopLeft: 'sm',
          px: 4, 
          py: 2, 
          mb: 2 
        })}>
          {bubble.messages.map((messageWithState, index) => {
            const { message, visibleToolCalls, stateAtMessage, stateBeforeMessage } = messageWithState;
            const agentMessage = message as AgentMessage;
            
            return (
              <div key={index}>
                {agentMessage.content && (
                  <div className={css({ 
                    '&:not(:first-child)': { mt: 2 },
                    '&:not(:last-child)': { mb: 2 }
                  })}>
                    <RoleplayText content={agentMessage.content} />
                  </div>
                )}
                
                {/* Tool presentations integrated within the same chat bubble */}
                {visibleToolCalls.map((toolCall, toolIndex) => (
                  <SpecialToolPresentation 
                    key={`${toolCall.tool_id}-${toolIndex}`} 
                    toolCall={toolCall} 
                    stateAtMessage={stateAtMessage}
                    stateBeforeMessage={stateBeforeMessage}
                  />
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function RoleplayPresenter({ messages, isStreamActive, agentState }: ConversationPresenterProps) {
  // Use demo data for development/testing, real messages for production
  // To use demo data: change 'messages' to 'demoMessages' below
  const messagesToUse = messages; // or demoMessages for testing
  
  // Build message bubbles by grouping consecutive messages
  const messageBubbles = useMemo(() => {
    const messagesWithState = buildMessagesWithState(messagesToUse, agentState as RoleplayState);
    return groupMessagesIntoBubbles(messagesWithState);
  }, [messagesToUse, agentState]);

  return (
    <div className={css({ 
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      maxWidth: '3xl',
      mx: 'auto'
    })}>
      {messageBubbles.map((bubble, bubbleIndex) => {
        if (bubble.role === 'user') {
          return (
            <UserBubble key={bubbleIndex} bubble={bubble} />
          );
        } else if (bubble.role === 'assistant') {
          return (
            <AgentBubble key={bubbleIndex} bubble={bubble} />
          );
        } else {
          return (
            <SystemMessage key={bubbleIndex} bubble={bubble} />
          );
        }
      })}
      
      {isStreamActive && (
        <div className={css({ color: 'gray.500' })}>
          <span className={css({ animation: 'pulse' })}>▋</span>
        </div>
      )}
    </div>
  );
}


function CharacterHeader({ character, stateAtMessage }: { character: CharacterState; stateAtMessage: RoleplayState }) {
  const moodEmoji = MOOD_EMOJIS[character.mood] || '😐';
  
  return (
    <div className={css({ 
      mb: 3
    })}>
      <div className={css({ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 2, 
        fontSize: 'sm' 
      })}>
        <span className={css({ 
          fontWeight: 'medium', 
          color: 'gray.300' 
        })}>{character.name}</span>
        <span className={css({ 
          fontSize: 'lg' 
        })}>{moodEmoji}</span>
        <span className={css({ 
          color: 'gray.500', 
          fontSize: 'xs' 
        })}>
          {character.mood} • {character.mood_intensity}
        </span>
      </div>
      
      {/* Scene context */}
      {stateAtMessage.global_scene && (
        <div className={css({ 
          fontSize: 'xs', 
          color: 'gray.500', 
          mt: 1, 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1 
        })}>
          <span>📍</span>
          <span>{stateAtMessage.global_scene.location}</span>
          {stateAtMessage.global_scene.atmosphere && (
            <>
              <span>•</span>
              <span>{stateAtMessage.global_scene.atmosphere}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function SpecialToolPresentation({ toolCall, stateBeforeMessage }: { toolCall: ToolCall; stateAtMessage: RoleplayState; stateBeforeMessage: RoleplayState }) {
  // Only handle agent tools here
  switch (toolCall.tool_name) {
    case 'set_mood':
      return <MoodTransition toolCall={toolCall} stateBeforeMessage={stateBeforeMessage} />;
    case 'character_action':
      return <CharacterAction toolCall={toolCall} />;
    case 'internal_thought':
      return <InternalThought toolCall={toolCall} />;
    default:
      return null;
  }
}

function CharacterAction({ toolCall }: { toolCall: ToolCall }) {
  const action = toolCall.parameters.action;
  if (!action) return null;
  
  return (
    <div className={css({ 
      color: 'blue.300', 
      fontStyle: 'italic', 
      mt: 2 
    })}>
      *{action}*
    </div>
  );
}

function MoodTransition({ toolCall, stateBeforeMessage }: { toolCall: ToolCall; stateBeforeMessage: RoleplayState }) {
  if (toolCall.type !== 'finished') return null;
  
  const newMood = toolCall.parameters.mood || 'neutral';
  
  // Get the previous mood from the character state BEFORE this tool executed
  const currentCharacter = stateBeforeMessage.current_character_id 
    ? stateBeforeMessage.characters[stateBeforeMessage.current_character_id] 
    : null;
  const oldMood = currentCharacter?.mood || 'neutral';
  
  const oldEmoji = MOOD_EMOJIS[oldMood] || '😐';
  const newEmoji = MOOD_EMOJIS[newMood] || '😐';
  
  return (
    <div className={css({ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 2, 
      color: 'gray.300', 
      mt: 2 
    })}>
      <span>{oldEmoji}</span>
      <span>→</span>
      <span>{newEmoji}</span>
      {toolCall.parameters.flavor_text && (
        <span className={css({ fontStyle: 'italic' })}>{toolCall.parameters.flavor_text}</span>
      )}
    </div>
  );
}

function InternalThought({ toolCall }: { toolCall: ToolCall }) {
  const thought = toolCall.parameters.thought;
  if (!thought) return null;
  
  return (
    <div className={css({ 
      display: 'flex', 
      alignItems: 'flex-start', 
      gap: 2, 
      color: 'yellow.300', 
      mt: 2 
    })}>
      <span>💭</span>
      <span className={css({ fontStyle: 'italic' })}>{thought}</span>
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
    <div className={css({ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 2, 
      color: 'purple.300', 
      mt: 2 
    })}>
      <span>📍</span>
      <span>{parts.join(' • ')}</span>
    </div>
  );
}

function SystemMessage({ bubble }: { bubble: MessageBubble }) {
  if (!bubble.systemTools || bubble.systemTools.length === 0) return null;
  
  return (
    <div className={css({ 
      mb: 4,
      display: 'flex',
      justifyContent: 'center'
    })}>
      <div className={css({ 
        px: 4,
        py: 2,
        bg: 'gray.900',
        border: '1px solid',
        borderColor: 'gray.700',
        rounded: 'md',
        fontSize: 'sm',
        color: 'gray.400',
        fontStyle: 'italic',
        maxWidth: 'lg',
        textAlign: 'center'
      })}>
        {bubble.systemTools.map((toolCall, index) => (
          <SystemToolPresentation 
            key={`${toolCall.tool_id}-${index}`} 
            toolCall={toolCall} 
          />
        ))}
      </div>
    </div>
  );
}

function SystemToolPresentation({ toolCall }: { toolCall: ToolCall }) {
  // Handle system tools
  switch (toolCall.tool_name) {
    case 'scene_setting':
      return <SceneSetting toolCall={toolCall} />;
    case 'remember_detail':
      return <div>💾 Memory stored: {toolCall.parameters.detail || 'Detail saved'}</div>;
    case 'correct_detail':
      return <div>✏️ Memory updated: {toolCall.parameters.correction || 'Detail corrected'}</div>;
    default:
      return null;
  }
}