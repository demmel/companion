import { AgentMessage, UserMessage } from '@/types';

export const demoMessages: (AgentMessage | UserMessage)[] = [
  { role: 'user' as const, content: [{ type: 'text', text: 'Hello! Can you roleplay as Elena?' }] },
  {
    role: 'assistant' as const,
    content: [],
    tool_calls: [{
      type: 'finished' as const,
      tool_name: 'assume_character',
      tool_id: 'call_1',
      parameters: {
        character_name: 'Elena',
        personality: 'Warm baker who loves making people smile',
        background: 'Runs a cozy bakery'
      },
      result: { type: 'success', content: 'Character created' }
    }]
  },
  {
    role: 'assistant' as const,
    content: [{ type: 'text', text: 'Good morning! Welcome to my little bakery!' }],
    tool_calls: [{
      type: 'finished' as const,
      tool_name: 'character_action',
      tool_id: 'call_2',
      parameters: { action: 'wipes flour-dusted hands on apron' },
      result: { type: 'success', content: 'Action performed' }
    }]
  },
  { role: 'user' as const, content: [{ type: 'text', text: 'The bakery smells amazing!' }] },
  { role: 'user' as const, content: [{ type: 'text', text: 'What do you recommend?' }] },
  {
    role: 'assistant' as const,
    content: [{ type: 'text', text: 'Oh, thank you so much! That just made my day!' }],
    tool_calls: [{
      type: 'finished' as const,
      tool_name: 'set_mood',
      tool_id: 'call_3',
      parameters: { mood: 'joyful', intensity: 'high', flavor_text: 'A customer complimented her baking!' },
      result: { type: 'success', content: 'Mood set' }
    }, {
      type: 'finished' as const,
      tool_name: 'internal_thought',
      tool_id: 'call_4',
      parameters: { thought: 'I love when customers appreciate the work I put into my recipes' },
      result: { type: 'success', content: 'Thought recorded' }
    }]
  },
  {
    role: 'assistant' as const,
    content: [{ type: 'text', text: 'I just pulled some blueberry muffins from the oven, and my grandmother\'s cinnamon rolls are always popular!' }],
    tool_calls: []
  },
  {
    role: 'assistant' as const,
    content: [],
    tool_calls: [{
      type: 'finished' as const,
      tool_name: 'scene_setting',
      tool_id: 'call_5',
      parameters: {
        location: 'Cozy neighborhood bakery',
        atmosphere: 'warm and inviting',
        time: 'early morning'
      },
      result: { type: 'success', content: 'Scene set' }
    }]
  },
  { role: 'user' as const, content: [{ type: 'text', text: 'I\'ll take both!' }] }
];