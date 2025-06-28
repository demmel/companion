import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useStreamProcessor } from '../useStreamProcessor';
import { AgentEvent } from '../../types';

describe('useStreamProcessor', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should initialize with empty state', () => {
    const { result } = renderHook(() => useStreamProcessor());
    
    expect(result.current.streamState).toEqual({
      items: [],
      toolMap: new Map(),
      insertionCounter: 0,
      isComplete: false
    });
  });

  it('should process text events and combine consecutive text', () => {
    const { result } = renderHook(() => useStreamProcessor(0)); // No batching delay
    
    // Add first text event
    act(() => {
      result.current.queueEvent({
        type: 'agent_text',
        content: 'Hello '
      });
    });

    expect(result.current.streamState.items).toHaveLength(1);
    expect(result.current.streamState.items[0]).toEqual({
      insertionOrder: 0,
      data: { type: 'text', content: 'Hello ' }
    });

    // Add second text event - should combine
    act(() => {
      result.current.queueEvent({
        type: 'agent_text',
        content: 'world!'
      });
    });

    expect(result.current.streamState.items).toHaveLength(1);
    expect(result.current.streamState.items[0]).toEqual({
      insertionOrder: 0,
      data: { type: 'text', content: 'Hello world!' }
    });
  });

  it('should not combine text events separated by tool events', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    // Text -> Tool -> Text should create separate text items
    act(() => {
      result.current.queueEvent({ type: 'agent_text', content: 'Before' });
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'test_tool',
        parameters: {}
      });
      result.current.queueEvent({ type: 'agent_text', content: 'After' });
    });

    expect(result.current.streamState.items).toHaveLength(3);
    expect(result.current.streamState.items[0].data).toEqual({ type: 'text', content: 'Before' });
    expect(result.current.streamState.items[1].data.type).toBe('tool');
    expect(result.current.streamState.items[2].data).toEqual({ type: 'text', content: 'After' });
  });

  it('should process tool started events correctly', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'search',
        parameters: { query: 'test' }
      });
    });

    expect(result.current.streamState.items).toHaveLength(1);
    expect(result.current.streamState.items[0]).toEqual({
      insertionOrder: 0,
      data: {
        type: 'tool',
        toolId: 'tool1',
        name: 'search',
        parameters: { query: 'test' },
        status: 'running'
      }
    });
    expect(result.current.streamState.toolMap.get('tool1')).toBe(0);
  });

  it('should update tool status on tool finished events', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    // Start tool
    act(() => {
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'search',
        parameters: { query: 'test' }
      });
    });

    // Finish tool successfully
    act(() => {
      result.current.queueEvent({
        type: 'tool_finished',
        tool_id: 'tool1',
        result_type: 'success',
        result: 'Found results'
      });
    });

    const toolData = result.current.streamState.items[0].data;
    expect(toolData.type).toBe('tool');
    if (toolData.type === 'tool') {
      expect(toolData.status).toBe('completed');
      expect(toolData.result).toBe('Found results');
      expect(toolData.error).toBeUndefined();
    }
  });

  it('should handle tool errors correctly', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'search',
        parameters: { query: 'test' }
      });
      result.current.queueEvent({
        type: 'tool_finished',
        tool_id: 'tool1',
        result_type: 'error',
        result: 'Connection failed'
      });
    });

    const toolData = result.current.streamState.items[0].data;
    if (toolData.type === 'tool') {
      expect(toolData.status).toBe('error');
      expect(toolData.error).toBe('Connection failed');
      expect(toolData.result).toBeUndefined();
    }
  });

  it('should handle multiple concurrent tools', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      // Start two tools
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'search',
        parameters: { query: 'test1' }
      });
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool2',
        tool_name: 'action',
        parameters: { action: 'jump' }
      });
      
      // Finish second tool first
      result.current.queueEvent({
        type: 'tool_finished',
        tool_id: 'tool2',
        result_type: 'success',
        result: 'Jumped successfully'
      });
      
      // Finish first tool
      result.current.queueEvent({
        type: 'tool_finished',
        tool_id: 'tool1',
        result_type: 'success',
        result: 'Search complete'
      });
    });

    expect(result.current.streamState.items).toHaveLength(2);
    
    // Check first tool
    const tool1Data = result.current.streamState.items[0].data;
    if (tool1Data.type === 'tool') {
      expect(tool1Data.toolId).toBe('tool1');
      expect(tool1Data.status).toBe('completed');
      expect(tool1Data.result).toBe('Search complete');
    }
    
    // Check second tool
    const tool2Data = result.current.streamState.items[1].data;
    if (tool2Data.type === 'tool') {
      expect(tool2Data.toolId).toBe('tool2');
      expect(tool2Data.status).toBe('completed');
      expect(tool2Data.result).toBe('Jumped successfully');
    }
  });

  it('should batch events with delay', async () => {
    const { result } = renderHook(() => useStreamProcessor(50));
    
    // Queue multiple events quickly
    act(() => {
      result.current.queueEvent({ type: 'agent_text', content: 'Hello' });
      result.current.queueEvent({ type: 'agent_text', content: ' world' });
      result.current.queueEvent({ type: 'agent_text', content: '!' });
    });

    // Should still be empty before batch processes
    expect(result.current.streamState.items).toHaveLength(0);
    
    // Advance timers to trigger batch processing
    act(() => {
      vi.advanceTimersByTime(50);
    });

    // Now should have combined text
    expect(result.current.streamState.items).toHaveLength(1);
    expect(result.current.streamState.items[0].data).toEqual({
      type: 'text',
      content: 'Hello world!'
    });
  });

  it('should handle response complete events', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      result.current.queueEvent({ type: 'agent_text', content: 'Done' });
      result.current.queueEvent({ type: 'response_complete' });
    });

    expect(result.current.streamState.isComplete).toBe(true);
  });

  it('should clear stream state', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    // Add some content
    act(() => {
      result.current.queueEvent({ type: 'agent_text', content: 'Test' });
    });

    expect(result.current.streamState.items).toHaveLength(1);
    
    // Clear stream
    act(() => {
      result.current.clearStream();
    });

    expect(result.current.streamState).toEqual({
      items: [],
      toolMap: new Map(),
      insertionCounter: 0,
      isComplete: false
    });
  });

  it('should handle agent error events', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      result.current.queueEvent({
        type: 'agent_error',
        message: 'Something went wrong',
        tool_name: 'broken_tool',
        tool_id: 'tool1'
      });
    });

    expect(result.current.streamState.items).toHaveLength(1);
    expect(result.current.streamState.items[0].data).toEqual({
      type: 'text',
      content: '❌ Error: Something went wrong'
    });
  });

  it('should maintain insertion order', () => {
    const { result } = renderHook(() => useStreamProcessor(0));
    
    act(() => {
      result.current.queueEvent({ type: 'agent_text', content: 'First' });
      result.current.queueEvent({
        type: 'tool_started',
        tool_id: 'tool1',
        tool_name: 'test',
        parameters: {}
      });
      result.current.queueEvent({ type: 'agent_text', content: 'Second' });
    });

    expect(result.current.streamState.items).toHaveLength(3);
    expect(result.current.streamState.items[0].insertionOrder).toBe(0);
    expect(result.current.streamState.items[1].insertionOrder).toBe(1);
    expect(result.current.streamState.items[2].insertionOrder).toBe(2);
  });
});