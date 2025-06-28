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
      { type: 'text', content: 'Hello ' },
      { type: 'text', content: 'there!' }
    ];
    
    const { result } = renderHook(() => useConversation(events));
    
    expect(result.current.messages).toEqual([
      { role: 'assistant', content: 'Hello there!', tool_calls: [] }
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it('should handle tool calls during streaming', () => {
    const events: AgentEvent[] = [
      { type: 'text', content: 'Let me help you. ' },
      { 
        type: 'tool_started', 
        tool_name: 'set_mood', 
        tool_id: 'call_1', 
        parameters: { mood: 'happy' } 
      },
      { 
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
      { type: 'text', content: 'Done!' }
    ];
    
    const { result, rerender } = renderHook((props) => useConversation(props.events), {
      initialProps: { events }
    });
    
    // Should be streaming
    expect(result.current.isStreamActive).toBe(true);
    
    // Add response_complete event
    events = [
      ...events,
      { type: 'response_complete' }
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
      { type: 'text', content: 'Working on it. ' },
      { 
        type: 'tool_started', 
        tool_name: 'tool_a', 
        tool_id: 'call_1', 
        parameters: { a: 1 } 
      },
      { 
        type: 'tool_started', 
        tool_name: 'tool_b', 
        tool_id: 'call_2', 
        parameters: { b: 2 } 
      },
      { 
        type: 'tool_finished', 
        tool_id: 'call_1', 
        result_type: 'success', 
        result: 'Result A' 
      },
      { 
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
});