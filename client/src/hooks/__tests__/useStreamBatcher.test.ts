import { renderHook, act } from '@testing-library/react';
import { useStreamBatcher } from '../useStreamBatcher';
import { AgentEvent } from '../../types';
import { vi } from 'vitest';

// Mock timers for testing batching behavior
vi.useFakeTimers();

describe('useStreamBatcher', () => {
  afterEach(() => {
    vi.clearAllTimers();
  });

  it('should start with empty events', () => {
    const { result } = renderHook(() => useStreamBatcher());
    
    expect(result.current.events).toEqual([]);
  });

  it('should batch events with default interval', () => {
    const { result } = renderHook(() => useStreamBatcher(50));
    
    const event1: AgentEvent = { id: 0, type: 'text', content: 'Hello' };
    const event2: AgentEvent = { id: 1, type: 'text', content: ' World' };

    act(() => {
      result.current.queueEvent(event1);
      result.current.queueEvent(event2);
    });
    
    // Events should not be visible yet (still batching)
    expect(result.current.events).toEqual([]);
    
    // Fast-forward timer
    act(() => {
      vi.advanceTimersByTime(50);
    });
    
    // Now events should be visible
    expect(result.current.events).toEqual([event1, event2]);
  });

  it('should process events immediately with zero interval', () => {
    const { result } = renderHook(() => useStreamBatcher(0));

    const event: AgentEvent = { id: 0, type: 'text', content: 'Immediate' };

    act(() => {
      result.current.queueEvent(event);
    });
    
    // Should be immediate
    expect(result.current.events).toEqual([event]);
  });

  it('should reset timer on new events', () => {
    const { result } = renderHook(() => useStreamBatcher(100));

    const event1: AgentEvent = { id: 0, type: 'text', content: 'First' };
    const event2: AgentEvent = { id: 1, type: 'text', content: 'Second' };

    act(() => {
      result.current.queueEvent(event1);
    });
    
    // Advance part way
    act(() => {
      vi.advanceTimersByTime(50);
    });
    
    // Add another event (should reset timer)
    act(() => {
      result.current.queueEvent(event2);
    });
    
    // Advance another 50ms (total 100ms from first event, but only 50ms from second)
    act(() => {
      vi.advanceTimersByTime(50);
    });
    
    // Events should not be visible yet
    expect(result.current.events).toEqual([]);
    
    // Advance the remaining 50ms
    act(() => {
      vi.advanceTimersByTime(50);
    });
    
    // Now both events should be visible
    expect(result.current.events).toEqual([event1, event2]);
  });

  it('should clear events', () => {
    const { result } = renderHook(() => useStreamBatcher(0));

    const event: AgentEvent = { id: 0, type: 'text', content: 'Test' };

    act(() => {
      result.current.queueEvent(event);
    });
    
    expect(result.current.events).toEqual([event]);
    
    act(() => {
      result.current.clearEvents();
    });
    
    expect(result.current.events).toEqual([]);
  });

  it('should handle multiple batches', () => {
    const { result } = renderHook(() => useStreamBatcher(50));

    const event1: AgentEvent = { id: 0, type: 'text', content: 'Batch 1' };
    const event2: AgentEvent = { id: 1, type: 'text', content: 'Batch 2' };

    // First batch
    act(() => {
      result.current.queueEvent(event1);
      vi.advanceTimersByTime(50);
    });
    
    expect(result.current.events).toEqual([event1]);
    
    // Second batch
    act(() => {
      result.current.queueEvent(event2);
      vi.advanceTimersByTime(50);
    });
    
    expect(result.current.events).toEqual([event1, event2]);
  });

  it('should clean up timers on unmount', () => {
    const { result, unmount } = renderHook(() => useStreamBatcher(100));
    
    act(() => {
      result.current.queueEvent({ id: 0, type: 'text', content: 'Test' });
    });
    
    // Unmount before timer fires
    unmount();
    
    // Should not cause any issues
    act(() => {
      vi.advanceTimersByTime(100);
    });
  });
});