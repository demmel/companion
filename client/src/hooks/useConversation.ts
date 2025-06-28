import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { AgentEvent, Message, UserMessage, ToolCall, ToolCallFinished } from '../types';
import { debug } from '@/utils/debug';

export interface UseConversationReturn {
  messages: Message[];
  isStreamActive: boolean;
  addUserMessage: (content: string) => void;
  loadConversation: (messages: Message[]) => void;
  clearConversation: () => void;
}

/**
 * Converts streaming events into structured messages that match the /conversation API format.
 * Shows the current agent response as it streams, updating it in real-time.
 */
export function useConversation(events: AgentEvent[]): UseConversationReturn {
  const [baseMessages, setBaseMessages] = useState<Message[]>([]);
  const [currentAgentResponse, setCurrentAgentResponse] = useState<{
    content: string;
    toolCalls: Map<string, ToolCall>;
  } | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const lastProcessedEventId = useRef<number | null>(null);

  useMemo(() => {
    debug.log('Base messages:', baseMessages);
  }, [baseMessages]);

  useMemo(() => {
    debug.log('Current agent response:', currentAgentResponse);
  }, [currentAgentResponse]);

  // Process events into current agent response
  useEffect(() => {
    if (events.length === 0) return;

    let content = currentAgentResponse?.content || '';
    const toolCalls = currentAgentResponse?.toolCalls || new Map<string, ToolCall>();
    let isComplete = false;

    for (const event of events) {
      if (lastProcessedEventId.current !== null && event.id <= lastProcessedEventId.current) {
        // Skip already processed events
        continue;
      }

      lastProcessedEventId.current = event.id;

      debug.log('Processing event:', event);

      switch (event.type) {
        case 'text':
          if (toolCalls.size > 0) {
            // If we have tool calls, finalize the current response first
            const contentToFinalize = content;
            const toolCallsToFinalize = Array.from(toolCalls.values());
            setBaseMessages(prev => [
              ...prev,
              {
                role: 'assistant',
                content: contentToFinalize,
                tool_calls: toolCallsToFinalize
              }
            ]);
            content = '';
            toolCalls.clear();
          }
          content += event.content;
          isComplete = false; // Still streaming
          break;
        case 'tool_started':
          toolCalls.set(event.tool_id, {
            tool_id: event.tool_id,
            tool_name: event.tool_name,
            type: 'started',
            parameters: event.parameters,
          });
          isComplete = false; // Still streaming
          break;
        case 'tool_finished':
          if (toolCalls.has(event.tool_id)) {
            let toolCall = toolCalls.get(event.tool_id);
            if (toolCall) {
              toolCall.type = 'finished';
              toolCall = toolCall as ToolCallFinished;
              toolCall.result = {
                type: event.result_type,
                content: event.result
              }
            }
          }
          isComplete = false; // Still streaming
          break;
        case 'response_complete':
          // Finalize the current agent respons
          if (content || toolCalls.size > 0) {
            const contentToFinalize = content;
            const toolCallsToFinalize = Array.from(toolCalls.values());
            setBaseMessages(prev => {
              return [
                ...prev,
                {
                  role: 'assistant',
                  content: contentToFinalize,
                  tool_calls: toolCallsToFinalize
                }
              ];
            });
          }
          content = '';
          toolCalls.clear();
          isComplete = true;
      }

      debug.log('Updated content:', content);
      debug.log('Updated tool calls:', Array.from(toolCalls.values()));
    }

    // Update the current agent response state
    if (content || toolCalls.size > 0) {
      setCurrentAgentResponse({
        content: content,
        toolCalls: toolCalls
      });
    } else {
      setCurrentAgentResponse(null);
    }

    setIsStreamActive(!isComplete); // If we have a complete response, stop streaming
  }, [events]);

  // Only reconstruct messages array when baseMessages or currentAgentResponse actually change
  const messages = useMemo(() => {
    if (!currentAgentResponse || currentAgentResponse.content === '' && currentAgentResponse.toolCalls.size === 0) {
      return baseMessages;
    }
    
    return [
      ...baseMessages,
      {
        role: 'assistant' as const,
        content: currentAgentResponse.content,
        tool_calls: Array.from(currentAgentResponse.toolCalls.values())
      }
    ];
  }, [baseMessages, currentAgentResponse]);

  const addUserMessage = useCallback((content: string) => {
    const userMessage: UserMessage = {
      role: 'user',
      content
    };
    setBaseMessages(prev => [...prev, userMessage]);
    setCurrentAgentResponse(null);
    setIsStreamActive(true); // Indicate that a new user message has been added
  }, []);

  const loadConversation = useCallback((conversationMessages: Message[]) => {
    setBaseMessages(conversationMessages);
    setCurrentAgentResponse(null);
    setIsStreamActive(false); // Reset stream state when loading a conversation
    lastProcessedEventId.current = null; // Reset event processing state
  }, []);

  const clearConversation = useCallback(() => {
    setBaseMessages([]);
    setCurrentAgentResponse(null);
    setIsStreamActive(false); // Reset stream state when clearing conversation
    lastProcessedEventId.current = null; // Reset event processing state
  }, []);

  return {
    messages,
    isStreamActive,
    addUserMessage,
    loadConversation,
    clearConversation
  };
}