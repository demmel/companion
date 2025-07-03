import { buildMessagesWithState, groupMessagesIntoBubbles } from '../RoleplayPresenter';
import { AgentMessage, UserMessage, ToolCall } from '@/types';

// Helper to create a finished tool call
function createToolCall(toolName: string, parameters: any, toolId = '1'): ToolCall {
  return {
    type: 'finished',
    tool_id: toolId,
    tool_name: toolName,
    parameters,
    result: { type: 'success', content: 'Success' }
  };
}

// Helper to create agent message
function createAgentMessage(content: string, toolCalls: ToolCall[] = []): AgentMessage {
  return {
    role: 'assistant',
    content,
    tool_calls: toolCalls
  };
}

// Helper to create user message
function createUserMessage(content: string): UserMessage {
  return {
    role: 'user',
    content
  };
}

describe('RoleplayPresenter Logic Functions', () => {
  describe('buildMessagesWithState', () => {
    test('character establishment - content comes before character assumption', () => {
      const messages = [
        createAgentMessage("Hello there!", [
          createToolCall('assume_character', {
            character_name: 'Alice',
            personality: 'friendly',
            background: 'a helpful assistant'
          })
        ])
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result).toHaveLength(1);
      // "Hello there!" is spoken BEFORE Alice is established, so no header
      expect(result[0].shouldShowHeader).toBe(false);
      // But Alice is now established for future messages
      expect(result[0].stateAtMessage.current_character_id).toBe('char_Alice');
      expect(result[0].currentCharacter?.name).toBe('Alice');
    });

    test('character speaks after being established', () => {
      const messages = [
        createAgentMessage("I'll become Alice", [
          createToolCall('assume_character', {
            character_name: 'Alice',
            personality: 'friendly'
          })
        ]),
        createAgentMessage("Hello, I'm Alice speaking now!")
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result).toHaveLength(2);
      // First message: spoken before Alice established
      expect(result[0].shouldShowHeader).toBe(false);
      expect(result[0].stateAtMessage.current_character_id).toBe('char_Alice');
      
      // Second message: Alice is now speaking (established in previous message)
      expect(result[1].shouldShowHeader).toBe(true);
      expect(result[1].currentCharacter?.name).toBe('Alice');
    });

    test('character switch mid-conversation', () => {
      const messages = [
        createAgentMessage("Setting up Alice", [
          createToolCall('assume_character', {
            character_name: 'Alice',
            personality: 'friendly'
          })
        ]),
        createAgentMessage("Hello from Alice!"),
        createUserMessage("Hi Alice, nice to meet you!"),
        createAgentMessage("Alice responds again"),
        createAgentMessage("Now switching to Bob", [
          createToolCall('assume_character', {
            character_name: 'Bob', 
            personality: 'gruff'
          })
        ]),
        createAgentMessage("Greetings from Bob!")
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result).toHaveLength(6);
      
      // Message 0: Setup Alice (no header - spoken before Alice established)
      expect(result[0].shouldShowHeader).toBe(false);
      expect(result[0].stateAtMessage.current_character_id).toBe('char_Alice');
      
      // Message 1: Alice speaking (show header - Alice was established)
      expect(result[1].shouldShowHeader).toBe(true);
      expect(result[1].currentCharacter?.name).toBe('Alice');
      
      // Message 2: User message (no character state)
      expect(result[2].message.role).toBe('user');
      
      // Message 3: Alice speaking again after user (show header - user reset speaking)
      expect(result[3].shouldShowHeader).toBe(true);
      expect(result[3].currentCharacter?.name).toBe('Alice');
      
      // Message 4: Switch to Bob (no header - spoken while Alice active, establishes Bob)
      expect(result[4].shouldShowHeader).toBe(false);
      expect(result[4].stateAtMessage.current_character_id).toBe('char_Bob');
      
      // Message 5: Bob speaking (show header - character changed from Alice to Bob)
      expect(result[5].shouldShowHeader).toBe(true);
      expect(result[5].currentCharacter?.name).toBe('Bob');
    });

    test('user message resets speaking character tracking', () => {
      const messages = [
        createAgentMessage("I'm Alice", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("Hello from Alice!"),
        createUserMessage("Hi Alice!"),
        createAgentMessage("Alice responds again!")
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result).toHaveLength(4);
      expect(result[1].shouldShowHeader).toBe(true);  // Alice first speaks
      expect(result[3].shouldShowHeader).toBe(true);  // Alice speaks again after user
    });

    test('empty content with assume_character only has no header', () => {
      const messages = [
        createAgentMessage("", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ])
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result[0].shouldShowHeader).toBe(false); // No visible content
      expect(result[0].stateAtMessage.current_character_id).toBe('char_Alice');
    });

    test('visible tool calls count as content for header decision', () => {
      const messages = [
        createAgentMessage("Setting up Alice", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("", [ // No text content, but has visible tool
          createToolCall('set_mood', { mood: 'happy' })
        ])
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result[1].shouldShowHeader).toBe(true); // set_mood is visible content
      expect(result[1].currentCharacter?.name).toBe('Alice');
    });

    test('mood state evolution', () => {
      const messages = [
        createAgentMessage("", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("I'm feeling good", [
          createToolCall('set_mood', { mood: 'happy', intensity: 'moderate' })
        ])
      ];

      const result = buildMessagesWithState(messages);
      
      expect(result[1].currentCharacter?.mood).toBe('happy');
      expect(result[1].currentCharacter?.mood_intensity).toBe('moderate');
    });
  });

  describe('groupMessagesIntoBubbles', () => {
    test('consecutive agent messages group together', () => {
      const messages = [
        createAgentMessage("Setting up", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("Hello!"),
        createAgentMessage("How are you?")
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      expect(bubbles).toHaveLength(1);
      expect(bubbles[0].role).toBe('assistant');
      expect(bubbles[0].messages).toHaveLength(3);
      // The last message that should show header wins
      expect(bubbles[0].shouldShowHeader).toBe(true);
      expect(bubbles[0].currentCharacter?.name).toBe('Alice');
    });

    test('user messages create separate bubbles', () => {
      const messages = [
        createUserMessage("Hello"),
        createUserMessage("How are you?"),
        createAgentMessage("I'm fine, thanks!")
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      expect(bubbles).toHaveLength(2);
      expect(bubbles[0].role).toBe('user');
      expect(bubbles[0].messages).toHaveLength(2);
      expect(bubbles[1].role).toBe('assistant');
      expect(bubbles[1].messages).toHaveLength(1);
    });

    test('system tools create separate system bubbles', () => {
      const messages = [
        createAgentMessage("Setting scene", [
          createToolCall('scene_setting', {
            location: 'forest',
            atmosphere: 'mysterious'
          })
        ]),
        createAgentMessage("Hello there!")
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      expect(bubbles).toHaveLength(2);
      expect(bubbles[0].role).toBe('system');
      expect(bubbles[0].systemTools).toHaveLength(1);
      expect(bubbles[0].systemTools![0].tool_name).toBe('scene_setting');
      expect(bubbles[1].role).toBe('assistant');
      expect(bubbles[1].messages).toHaveLength(2);
    });

    test('mixed system and agent tools preserve order by splitting', () => {
      const messages = [
        createAgentMessage("Complex message", [
          createToolCall('scene_setting', { location: 'forest' }, 'tool1'),
          createToolCall('set_mood', { mood: 'excited' }, 'tool2'),
          createToolCall('character_action', { action: 'waves' }, 'tool3')
        ])
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      expect(bubbles).toHaveLength(2);
      
      // First bubble: system tool (scene_setting)
      expect(bubbles[0].role).toBe('system');
      expect(bubbles[0].systemTools).toHaveLength(1);
      expect(bubbles[0].systemTools![0].tool_name).toBe('scene_setting');
      
      // Second bubble: agent message with agent tools
      expect(bubbles[1].role).toBe('assistant');
      expect(bubbles[1].messages[0].visibleToolCalls).toHaveLength(2);
      expect(bubbles[1].messages[0].visibleToolCalls.map(tc => tc.tool_name))
        .toEqual(['set_mood', 'character_action']);
    });

    test('character change within grouped messages - header behavior', () => {
      // This tests the potential bug we discussed
      const messages = [
        createAgentMessage("Hello", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("I'm Alice speaking", [
          createToolCall('assume_character', { character_name: 'Bob' })
        ]),
        createAgentMessage("Now I'm Bob")
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      expect(bubbles).toHaveLength(1);
      
      // Check individual message states are computed correctly
      expect(messagesWithState[0].shouldShowHeader).toBe(false); // Setup Alice
      expect(messagesWithState[0].currentCharacter?.name).toBe('Alice');
      
      expect(messagesWithState[1].shouldShowHeader).toBe(true);  // Alice speaking
      expect(messagesWithState[1].currentCharacter?.name).toBe('Alice');
      
      expect(messagesWithState[2].shouldShowHeader).toBe(true);  // Bob speaking  
      expect(messagesWithState[2].currentCharacter?.name).toBe('Bob');
      
      // But the bubble shows the last character that triggered a header
      expect(bubbles[0].shouldShowHeader).toBe(true);
      expect(bubbles[0].currentCharacter?.name).toBe('Bob'); // Last character wins
      
      // This means Alice's "I'm Alice speaking" gets misattributed to Bob's header!
    });

    test('empty messages with only hidden tools get filtered out', () => {
      const messages = [
        createAgentMessage("", [
          createToolCall('assume_character', { character_name: 'Alice' })
        ]),
        createAgentMessage("Hello!")
      ];

      const messagesWithState = buildMessagesWithState(messages);
      const bubbles = groupMessagesIntoBubbles(messagesWithState);
      
      // First message has no content and only hidden tools, should still be included
      // because it might set up state for the next message
      expect(bubbles).toHaveLength(1);
      expect(bubbles[0].messages).toHaveLength(2);
    });
  });

  describe('Tool categorization and ordering', () => {
    test('tools are categorized correctly', () => {
      const messages = [
        createAgentMessage("Testing tools", [
          createToolCall('assume_character', { character_name: 'Alice' }), // HIDDEN
          createToolCall('scene_setting', { location: 'park' }),           // SYSTEM
          createToolCall('set_mood', { mood: 'happy' }),                   // AGENT
          createToolCall('remember_detail', { detail: 'test' }),           // SYSTEM  
          createToolCall('character_action', { action: 'waves' }),         // AGENT
          createToolCall('internal_thought', { thought: 'hmm' })           // AGENT
        ])
      ];

      const messagesWithState = buildMessagesWithState(messages);
      
      // Check visible tool calls (should exclude assume_character)
      expect(messagesWithState[0].visibleToolCalls).toHaveLength(5);
      expect(messagesWithState[0].visibleToolCalls.map(tc => tc.tool_name))
        .toEqual(['scene_setting', 'set_mood', 'remember_detail', 'character_action', 'internal_thought']);
    });
  });
});