import { renderHook, act } from '@testing-library/react';
import { useConversation } from '../useConversation';
import { AgentEvent, UserMessage, AgentMessage } from '../../types';

describe('useConversation', () => {
  it('should start with empty messages', () => {
    const { result } = renderHook(() => useConversation([]));
    
    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it('should add user messages', () => {
    const { result } = renderHook(() => useConversation([]));
    
    act(() => {
      result.current.addUserMessage('Hello');
    });
    
    expect(result.current.messages).toEqual([
      { role: 'user', content: 'Hello' }
    ]);
  });

  it('should show streaming agent response as it comes', () => {
    const events: AgentEvent[] = [
      { id: 0, type: 'text', content: 'Hello ' },
      { id: 1, type: 'text', content: 'there!' }
    ];
    
    const { result } = renderHook(() => useConversation(events));
    
    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello there!', tool_calls: [] }
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it('should handle tool calls during streaming', () => {
    const events: AgentEvent[] = [
      { id: 0, type: 'text', content: 'Let me help you. ' },
      { 
        id: 1,
        type: 'tool_started', 
        tool_name: 'set_mood', 
        tool_id: 'call_1', 
        parameters: { mood: 'happy' } 
      },
      { 
        id: 2,
        type: 'tool_finished', 
        tool_id: 'call_1', 
        result_type: 'success', 
        result: 'Mood set to happy' 
      }
    ];
    
    const { result } = renderHook(() => useConversation(events));
    
    expect(result.current.messages).toEqual([
      {
        role: 'assistant',
        content: 'Let me help you. ',
        tool_calls: [
          {
            type: 'finished',
            tool_name: 'set_mood',
            tool_id: 'call_1',
            parameters: { mood: 'happy' },
            result: {
              type: 'success',
              content: 'Mood set to happy'
            }
          }
        ]
      }
    ]);
  });

  it('should show running tool calls before they finish', () => {
    const events: AgentEvent[] = [
      { 
        id: 0,
        type: 'tool_started', 
        tool_name: 'slow_tool', 
        tool_id: 'call_1', 
        parameters: { task: 'processing' } 
      }
    ];
    
    const { result } = renderHook(() => useConversation(events));
    
    expect(result.current.messages).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            type: 'started',
            tool_name: 'slow_tool',
            tool_id: 'call_1',
            parameters: { task: 'processing' }
          }
        ]
      }
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it('should finalize message on response_complete', () => {
    let events: AgentEvent[] = [
      { id: 0, type: 'text', content: 'Done!' }
    ];
    
    const { result, rerender } = renderHook((props) => useConversation(props.events), {
      initialProps: { events }
    });
    
    // Should be streaming
    expect(result.current.isStreamActive).toBe(true);
    
    // Add response_complete event
    events = [
      ...events,
      { id: 1, type: 'response_complete' }
    ];
    
    rerender({ events });
    
    expect(result.current.isStreamActive).toBe(false);
    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Done!', tool_calls: [] }
    ]);
  });

  it('should load conversation from API', () => {
    const { result } = renderHook(() => useConversation([]));
    
    const conversationMessages: [UserMessage, AgentMessage] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!', tool_calls: [] }
    ];
    
    act(() => {
      result.current.loadConversation(conversationMessages);
    });
    
    expect(result.current.messages).toEqual(conversationMessages);
  });

  it('should clear conversation', () => {
    const { result } = renderHook(() => useConversation([]));
    
    act(() => {
      result.current.addUserMessage('Hello');
      result.current.clearConversation();
    });
    
    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it('should handle multiple tool calls', () => {
    const events: AgentEvent[] = [
      { id: 0, type: 'text', content: 'Working on it. ' },
      { 
        id: 1,
        type: 'tool_started', 
        tool_name: 'tool_a', 
        tool_id: 'call_1', 
        parameters: { a: 1 } 
      },
      { 
        id: 2,
        type: 'tool_started', 
        tool_name: 'tool_b', 
        tool_id: 'call_2', 
        parameters: { b: 2 } 
      },
      { 
        id: 3,
        type: 'tool_finished', 
        tool_id: 'call_1', 
        result_type: 'success', 
        result: 'Result A' 
      },
      { 
        id: 4,
        type: 'tool_finished', 
        tool_id: 'call_2', 
        result_type: 'error', 
        result: 'Error B' 
      }
    ];
    
    const { result } = renderHook(() => useConversation(events));
    
    const message = result.current.messages[0] as AgentMessage;
    expect(message.tool_calls).toHaveLength(2);
    
    const toolA = message.tool_calls.find(t => t.tool_id === 'call_1');
    const toolB = message.tool_calls.find(t => t.tool_id === 'call_2');
    
    expect(toolA).toEqual({
      type: 'finished',
      tool_name: 'tool_a',
      tool_id: 'call_1',
      parameters: { a: 1 },
      result: { type: 'success', content: 'Result A' }
    });
    
    expect(toolB).toEqual({
      type: 'finished',
      tool_name: 'tool_b',
      tool_id: 'call_2',
      parameters: { b: 2 },
      result: { type: 'error', content: 'Error B' }
    });
  });

  it('should not duplicate messages', () => {
    let events: AgentEvent[] = [
      { id: 0, type: 'text', content: 'Hello' },
      { id: 1, type: 'text', content: ' World' }
    ];
    
    const { result, rerender } = renderHook(({ events }) => useConversation(events), {
      initialProps: { events }
    });
    
    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World', tool_calls: [] }
    ]);

    // Add another event that should not duplicate
    events = [
      ...events,
      { id: 2, type: 'text', content: '!' },
      { id: 3, type: 'response_complete' }
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World!', tool_calls: [] }
    ]);

    // Add a new user message
    act(() => {
      result.current.addUserMessage('New message');
    });

    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World!', tool_calls: [] },
      { role: 'user', content: 'New message' }
    ]);

    // Add another agent message
    events = [
      ...events,
      { id: 4, type: 'text', content: 'Another response' },
      { id: 5, type: 'response_complete' }
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World!', tool_calls: [] },
      { role: 'user', content: 'New message' },
      { role: 'assistant', content: 'Another response', tool_calls: [] }
    ]);

    // Add a new user message again
    act(() => {
      result.current.addUserMessage('Another user message');
    });

    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World!', tool_calls: [] },
      { role: 'user', content: 'New message' },
      { role: 'assistant', content: 'Another response', tool_calls: [] },
      { role: 'user', content: 'Another user message' }
    ]);

    // Add tool calls
    events = [
      ...events,
      { 
        id: 6,
        type: 'tool_started', 
        tool_name: 'example_tool', 
        tool_id: 'call_3', 
        parameters: { param: 'value' } 
      },
      { 
        id: 7,
        type: 'tool_finished', 
        tool_id: 'call_3', 
        result_type: 'success', 
        result: 'Tool call result' 
      },
      { id: 8, type: 'text', content: 'Final response' },
      { id: 9, type: 'response_complete' }
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello World!', tool_calls: [] },
      { role: 'user', content: 'New message' },
      { role: 'assistant', content: 'Another response', tool_calls: [] },
      { role: 'user', content: 'Another user message' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            type: 'finished',
            tool_name: 'example_tool',
            tool_id: 'call_3',
            parameters: { param: 'value' },
            result: { type: 'success', content: 'Tool call result' }
          }
        ]
      },
      {
        role: 'assistant',
        content: 'Final response',
        tool_calls: []
      }
    ]);
  });
});