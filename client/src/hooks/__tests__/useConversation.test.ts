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

  it('should handle summarization events correctly', () => {
    const events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 10,
        recent_messages_kept: 6,
        context_usage_before: 85.5
      },
      {
        id: 1,
        type: 'summarization_finished',
        summary: 'User and assistant discussed various topics including weather, food preferences, and travel plans.',
        messages_summarized: 10,
        messages_after: 7,
        context_usage_after: 42.3
      }
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe('system');
    
    const content = result.current.messages[0].content;
    expect(typeof content).toBe('object');
    expect((content as any).type).toBe('summarization');
    expect((content as any).title).toContain('✅ Summarized 10 messages');
    expect((content as any).summary).toBe('User and assistant discussed various topics including weather, food preferences, and travel plans.');
    expect((content as any).messages_summarized).toBe(10);
    expect((content as any).context_usage_before).toBe(85.5);
    expect((content as any).context_usage_after).toBe(42.3);
  });

  it('should show summarization progress during streaming', () => {
    const events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 8,
        recent_messages_kept: 4,
        context_usage_before: 78.2
      }
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe('system');
    expect(result.current.messages[0].content).toBe('📝 Summarizing 8 older messages to manage context (78.2% usage)...');
    expect(result.current.isStreamActive).toBe(true);
  });

  it('should complete summarization and stop streaming', () => {
    let events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 5,
        recent_messages_kept: 3,
        context_usage_before: 90.1
      }
    ];

    const { result, rerender } = renderHook(({ events }) => useConversation(events), {
      initialProps: { events }
    });

    expect(result.current.isStreamActive).toBe(true);

    // Complete the summarization
    events = [
      ...events,
      {
        id: 1,
        type: 'summarization_finished',
        summary: 'Previous conversation about project planning and deadlines.',
        messages_summarized: 5,
        messages_after: 4,
        context_usage_after: 35.7
      }
    ];

    rerender({ events });

    expect(result.current.isStreamActive).toBe(true); // Still streaming until response_complete
    expect(result.current.messages).toHaveLength(1);
    
    const content = result.current.messages[0].content;
    expect((content as any).type).toBe('summarization');
    expect((content as any).context_usage_before).toBe(90.1);
    expect((content as any).context_usage_after).toBe(35.7);
  });

  it('should handle summarization followed by regular response', () => {
    const events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 6,
        recent_messages_kept: 4,
        context_usage_before: 82.0
      },
      {
        id: 1,
        type: 'summarization_finished',
        summary: 'Discussion about travel destinations and budget planning.',
        messages_summarized: 6,
        messages_after: 5,
        context_usage_after: 38.5
      },
      { id: 2, type: 'text', content: 'Now I can continue helping you!' },
      { id: 3, type: 'response_complete' }
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(2);
    
    // First message: summarization
    expect(result.current.messages[0].role).toBe('system');
    expect((result.current.messages[0].content as any).type).toBe('summarization');
    
    // Second message: regular assistant response
    expect(result.current.messages[1].role).toBe('assistant');
    expect(result.current.messages[1].content).toBe('Now I can continue helping you!');
    expect(result.current.isStreamActive).toBe(false);
  });

  it('should maintain correct role for system messages through response_complete', () => {
    const events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 5,
        recent_messages_kept: 3,
        context_usage_before: 85.0
      },
      {
        id: 1,
        type: 'summarization_finished',
        summary: 'Test summary content',
        messages_summarized: 5,
        messages_after: 4,
        context_usage_after: 40.0
      },
      { id: 2, type: 'response_complete' }
    ];

    const { result } = renderHook(() => useConversation(events));


    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe('system'); // This should NOT be 'assistant'
    expect(result.current.messages[0]).not.toHaveProperty('tool_calls');
    expect(result.current.isStreamActive).toBe(false);
  });

  it('should not create AgentMessage with SummarizationContent when text events follow summarization', () => {
    // This test covers the specific bug where text events after summarization_finished
    // would reset role to 'assistant' but leave SummarizationContent as content,
    // causing agentMessage.content.trim() to fail in RoleplayPresenter
    const events: AgentEvent[] = [
      {
        id: 0,
        type: 'summarization_started',
        messages_to_summarize: 6,
        recent_messages_kept: 4,
        context_usage_before: 75.0
      },
      {
        id: 1,
        type: 'summarization_finished',
        summary: 'Previous conversation about exploring nature and mushrooms.',
        messages_summarized: 6,
        messages_after: 8,
        context_usage_after: 68.0
      },
      // Text events that follow summarization (this was causing the bug)
      { id: 2, type: 'text', content: '*Eyes light up*' },
      { id: 3, type: 'text', content: ' Amazing facts!' },
      { id: 4, type: 'response_complete' }
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(2);
    
    // First message: properly created system message with SummarizationContent
    const systemMessage = result.current.messages[0];
    expect(systemMessage.role).toBe('system');
    expect(typeof systemMessage.content).toBe('object');
    expect((systemMessage.content as any).type).toBe('summarization');
    expect((systemMessage.content as any).summary).toBe('Previous conversation about exploring nature and mushrooms.');
    expect(systemMessage).not.toHaveProperty('tool_calls');
    
    // Second message: properly created assistant message with string content
    const assistantMessage = result.current.messages[1];
    expect(assistantMessage.role).toBe('assistant');
    expect(typeof assistantMessage.content).toBe('string'); // This should be a string, NOT SummarizationContent
    expect(assistantMessage.content).toBe('*Eyes light up* Amazing facts!');
    expect(assistantMessage).toHaveProperty('tool_calls');
    expect((assistantMessage as any).tool_calls).toEqual([]);
    
    // Verify the string content can be trimmed (this would fail with the bug)
    expect(() => {
      const content = assistantMessage.content as string;
      return content.trim();
    }).not.toThrow();
    
    expect(result.current.isStreamActive).toBe(false);
  });
});