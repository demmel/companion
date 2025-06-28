import { useState, useRef, useCallback, useEffect } from 'react';
import { AgentEvent, Message } from '../types';

type StreamData = 
  | { type: 'text'; content: string; role: 'user' | 'agent' }
  | { 
      type: 'tool'; 
      toolId: string; 
      name: string; 
      parameters: Record<string, any>; 
      status: 'running' | 'completed' | 'error';
      result?: string;
      error?: string;
    };

interface StreamItem {
  insertionOrder: number;
  data: StreamData;
}

interface StreamState {
  items: StreamItem[];
  toolMap: Map<string, number>; // toolId -> array index
  insertionCounter: number;
  isComplete: boolean;
}

function processEvent(state: StreamState, event: AgentEvent): StreamState {
  switch (event.type) {
    case 'text': {
      const lastItem = state.items[state.items.length - 1];
      const currentRole = 'agent'; // Server text events are always from agent
      
      // Combine with previous text item if same role
      if (lastItem?.data.type === 'text' && lastItem.data.role === currentRole) {
        return {
          ...state,
          items: [
            ...state.items.slice(0, -1),
            {
              ...lastItem,
              data: {
                type: 'text',
                content: lastItem.data.content + event.content,
                role: currentRole
              }
            }
          ]
        };
      }
      
      // Create new text item
      return {
        ...state,
        items: [
          ...state.items,
          {
            insertionOrder: state.insertionCounter,
            data: { type: 'text', content: event.content, role: currentRole }
          }
        ],
        insertionCounter: state.insertionCounter + 1
      };
    }

    case 'tool_started': {
      const newIndex = state.items.length;
      const toolItem: StreamItem = {
        insertionOrder: state.insertionCounter,
        data: {
          type: 'tool',
          toolId: event.tool_id,
          name: event.tool_name,
          parameters: event.parameters,
          status: 'running'
        }
      };

      const newToolMap = new Map(state.toolMap);
      newToolMap.set(event.tool_id, newIndex);
      
      return {
        ...state,
        items: [...state.items, toolItem],
        toolMap: newToolMap,
        insertionCounter: state.insertionCounter + 1
      };
    }

    case 'tool_finished': {
      const itemIndex = state.toolMap.get(event.tool_id);
      if (itemIndex === undefined) return state; // Tool not found
      
      return {
        ...state,
        items: state.items.map((item, index) => 
          index === itemIndex ? {
            ...item,
            data: {
              ...item.data as Extract<StreamData, { type: 'tool' }>,
              status: event.result_type === 'success' ? 'completed' : 'error',
              result: event.result_type === 'success' ? event.result : undefined,
              error: event.result_type === 'error' ? event.result : undefined
            }
          } : item
        )
      };
    }

    case 'error': {
      // Add error as a special text item
      return {
        ...state,
        items: [
          ...state.items,
          {
            insertionOrder: state.insertionCounter,
            data: { type: 'text', content: `❌ Error: ${event.message}` }
          }
        ],
        insertionCounter: state.insertionCounter + 1
      };
    }

    case 'response_complete': {
      return {
        ...state,
        isComplete: true
      };
    }

    default:
      return state;
  }
}

export interface UseStreamProcessorReturn {
  streamState: StreamState;
  queueEvent: (event: AgentEvent) => void;
  queueUserMessage: (content: string) => void;
  loadConversation: (messages: Message[]) => void;
  clearStream: () => void;
}

export function useStreamProcessor(batchInterval: number = 50): UseStreamProcessorReturn {
  const eventBufferRef = useRef<AgentEvent[]>([]);
  const batchTimerRef = useRef<number>();
  
  const [streamState, setStreamState] = useState<StreamState>({
    items: [],
    toolMap: new Map(),
    insertionCounter: 0,
    isComplete: false
  });

  const processBatch = useCallback(() => {
    if (eventBufferRef.current.length === 0) return;
    
    const events = eventBufferRef.current.splice(0); // Clear buffer
    
    setStreamState(prevState => {
      let newState = { ...prevState };
      
      for (const event of events) {
        newState = processEvent(newState, event);
      }
      
      return newState;
    });
  }, []);

  const queueEvent = useCallback((event: AgentEvent) => {
    eventBufferRef.current.push(event);
    
    // Reset batch timer
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
    }
    
    if (batchInterval === 0) {
      // Process immediately
      processBatch();
    } else {
      batchTimerRef.current = setTimeout(processBatch, batchInterval);
    }
  }, [processBatch, batchInterval]);

  const queueUserMessage = useCallback((content: string) => {
    setStreamState(prevState => ({
      ...prevState,
      items: [
        ...prevState.items,
        {
          insertionOrder: prevState.insertionCounter,
          data: { type: 'text', content, role: 'user' }
        }
      ],
      insertionCounter: prevState.insertionCounter + 1,
      isComplete: false
    }));
  }, []);

  const loadConversation = useCallback((messages: Message[]) => {
    const items: StreamItem[] = messages.map((message, index) => ({
      insertionOrder: index,
      data: {
        type: 'text',
        content: message.content,
        role: message.role === 'user' ? 'user' : 'agent'
      }
    }));

    setStreamState({
      items,
      toolMap: new Map(),
      insertionCounter: messages.length,
      isComplete: true
    });
  }, []);

  const clearStream = useCallback(() => {
    eventBufferRef.current = [];
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
    }
    setStreamState({
      items: [],
      toolMap: new Map(),
      insertionCounter: 0,
      isComplete: false
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (batchTimerRef.current) {
        clearTimeout(batchTimerRef.current);
      }
    };
  }, []);

  return {
    streamState,
    queueEvent,
    queueUserMessage,
    loadConversation,
    clearStream
  };
}