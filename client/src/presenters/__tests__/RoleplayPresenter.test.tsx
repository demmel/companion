import { render, screen } from '@testing-library/react';
import { RoleplayPresenter } from '../RoleplayPresenter';
import { UserMessage, AgentMessage, ToolCallFinished } from '../../types';

describe('RoleplayPresenter', () => {
  const mockAgentState = {
    current_character_id: null,
    characters: {},
    global_scene: null,
    global_memories: []
  };

  it('should show generic agent header when no character is active', () => {
    const messages: UserMessage[] = [
      { role: 'user', content: 'Hello' }
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('should hide assume_character tool and not show header until character speaks', () => {
    const assumeCharacterTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'assume_character',
      tool_id: 'call_1',
      parameters: {
        character_name: 'Bob',
        personality: 'friendly and curious'
      },
      result: {
        type: 'success',
        content: 'Character created'
      }
    };

    const messages = [
      { role: 'user', content: 'Can you play as Bob?' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [assumeCharacterTool]
      } as AgentMessage,
      { role: 'user', content: 'Say hello' },
      {
        role: 'assistant',
        content: 'Hello there!',
        tool_calls: []
      } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should NOT show the tool call
    expect(screen.queryByText('assume_character')).not.toBeInTheDocument();
    
    // Should show character header when they speak
    expect(screen.getByText('🎭')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
    expect(screen.getByText('😐')).toBeInTheDocument(); // neutral mood emoji
    
    // Should show dialogue
    expect(screen.getByText('Hello there!')).toBeInTheDocument();
  });

  it('should track character state evolution over conversation', () => {
    const assumeTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'assume_character',
      tool_id: 'call_1',
      parameters: {
        character_name: 'Alice',
        personality: 'mysterious'
      },
      result: { type: 'success', content: 'Character created' }
    };

    const moodTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'set_mood',
      tool_id: 'call_2',
      parameters: {
        mood: 'happy',
        intensity: 'high'
      },
      result: { type: 'success', content: 'Mood set' }
    };

    const messages = [
      { role: 'user', content: 'Play as Alice' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [assumeTool]
      } as AgentMessage,
      { role: 'user', content: 'Be happy' },
      {
        role: 'assistant',
        content: 'I feel great!',
        tool_calls: [moodTool]
      } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should show Alice in the character header
    expect(screen.getByText('Alice')).toBeInTheDocument();
    
    // Should show happy mood in final message
    expect(screen.getByText('😊')).toBeInTheDocument(); // happy emoji
    expect(screen.getByText('(happy - high)')).toBeInTheDocument();
    
    // Should show the dialogue content
    expect(screen.getByText('I feel great!')).toBeInTheDocument();
  });

  it('should hide memory tools', () => {
    const hiddenTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'remember_detail',
      tool_id: 'call_1',
      parameters: {
        detail: 'User likes coffee'
      },
      result: { type: 'success', content: 'Memory stored' }
    };

    const messages = [
      { role: 'user', content: 'I like coffee' },
      {
        role: 'assistant',
        content: 'Got it!',
        tool_calls: [hiddenTool]
      } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should NOT show hidden tool
    expect(screen.queryByText('remember_detail')).not.toBeInTheDocument();
    expect(screen.queryByText('Memory stored')).not.toBeInTheDocument();
    
    // Should show dialogue
    expect(screen.getByText('Got it!')).toBeInTheDocument();
  });

  it('should show special presentations for roleplay tools', () => {
    const actionTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'character_action',
      tool_id: 'call_1',
      parameters: {
        action: 'waves enthusiastically'
      },
      result: { type: 'success', content: 'Action performed' }
    };

    const thoughtTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'internal_thought',
      tool_id: 'call_2',
      parameters: {
        thought: 'This person seems nice'
      },
      result: { type: 'success', content: 'Thought recorded' }
    };

    const messages = [
      { role: 'user', content: 'Hello' },
      {
        role: 'assistant',
        content: 'Hi there!',
        tool_calls: [actionTool, thoughtTool]
      } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should show action in italics with asterisks
    expect(screen.getByText('*waves enthusiastically*')).toBeInTheDocument();
    
    // Should show thought with emoji
    expect(screen.getByText('💭 This person seems nice')).toBeInTheDocument();
    
    // Should show dialogue
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('should only show character header when character changes', () => {
    const assumeBob: ToolCallFinished = {
      type: 'finished',
      tool_name: 'assume_character',
      tool_id: 'call_1',
      parameters: { character_name: 'Bob', personality: 'friendly' },
      result: { type: 'success', content: 'Character created' }
    };

    const assumeAlice: ToolCallFinished = {
      type: 'finished',
      tool_name: 'assume_character',
      tool_id: 'call_2',
      parameters: { character_name: 'Alice', personality: 'mysterious' },
      result: { type: 'success', content: 'Character created' }
    };

    const messages = [
      { role: 'user', content: 'Play as Bob' },
      { role: 'assistant', content: '', tool_calls: [assumeBob] } as AgentMessage,
      { role: 'user', content: 'Say hi' },
      { role: 'assistant', content: 'Hello!', tool_calls: [] } as AgentMessage,
      { role: 'user', content: 'Now play as Alice' },
      { role: 'assistant', content: '', tool_calls: [assumeAlice] } as AgentMessage,
      { role: 'user', content: 'Say something mysterious' },
      { role: 'assistant', content: 'The shadows whisper secrets...', tool_calls: [] } as AgentMessage,
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should show both character names (headers only appear when character changes)
    expect(screen.getByText('Bob')).toBeInTheDocument();
    expect(screen.getByText('Alice')).toBeInTheDocument();
    
    // Should show both dialogues
    expect(screen.getByText('Hello!')).toBeInTheDocument();
    expect(screen.getByText('The shadows whisper secrets...')).toBeInTheDocument();
  });

  it('should show scene setting', () => {
    const sceneTool: ToolCallFinished = {
      type: 'finished',
      tool_name: 'scene_setting',
      tool_id: 'call_1',
      parameters: {
        location: 'Dark alley',
        atmosphere: 'mysterious',
        time: 'midnight'
      },
      result: { type: 'success', content: 'Scene set' }
    };

    const messages = [
      { role: 'user', content: 'Set the scene' },
      {
        role: 'assistant',
        content: 'The scene is set.',
        tool_calls: [sceneTool]
      } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={false} 
        agentState={mockAgentState}
      />
    );

    // Should show scene setting
    expect(screen.getByText('📍 Dark alley - mysterious - (midnight)')).toBeInTheDocument();
    
    // Should show dialogue
    expect(screen.getByText('The scene is set.')).toBeInTheDocument();
  });

  it('should show streaming cursor when active', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!', tool_calls: [] } as AgentMessage
    ];

    render(
      <RoleplayPresenter 
        messages={messages} 
        isStreamActive={true} 
        agentState={mockAgentState}
      />
    );

    expect(screen.getByText('▋')).toBeInTheDocument();
  });
});