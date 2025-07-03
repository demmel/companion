import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { AgentEvent, Message, UserMessage, SystemMessage, SystemContent, SummarizationContent, ToolCall, ToolCallFinished } from '../types';
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
  const [currentResponse, setCurrentResponse] = useState<{
    role: 'assistant' | 'system';
    content: string | SystemContent;
    toolCalls: Map<string, ToolCall>;
    summarizationData?: {
      messages_to_summarize: number;
      context_usage_before: number;
      messages_summarized?: number;
      context_usage_after?: number;
      summary?: string;
    };
  } | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const lastProcessedEventId = useRef<number | null>(null);

  useMemo(() => {
    debug.log('Base messages:', baseMessages);
  }, [baseMessages]);

  useMemo(() => {
    debug.log('Current response:', currentResponse);
  }, [currentResponse]);

  // Process events into current agent response
  useEffect(() => {
    if (events.length === 0) return;

    let content = currentResponse?.content || '';
    let role: 'assistant' | 'system' = currentResponse?.role || 'assistant';
    const toolCalls = currentResponse?.toolCalls || new Map<string, ToolCall>();
    let summarizationData = currentResponse?.summarizationData;
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
          if (toolCalls.size > 0 || (role === 'system' && content)) {
            // If we have tool calls or a system message, finalize the current response first
            const contentToFinalize = content;
            const toolCallsToFinalize = Array.from(toolCalls.values());
            setBaseMessages(prev => [
              ...prev,
              {
                role: role,
                content: contentToFinalize,
                tool_calls: role === 'assistant' ? toolCallsToFinalize : []
              } as Message
            ]);
            content = '';
            toolCalls.clear();
          }
          role = 'assistant'; // Text events are always assistant responses
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
        case 'summarization_started':
          // Start a new system message response
          role = 'system';
          content = `📝 Summarizing ${event.messages_to_summarize} older messages to manage context (${event.context_usage_before.toFixed(1)}% usage)...`;
          toolCalls.clear();
          summarizationData = {
            messages_to_summarize: event.messages_to_summarize,
            context_usage_before: event.context_usage_before,
          };
          isComplete = false;
          break;
        case 'summarization_finished':
          // Update the system message with structured summarization content
          if (summarizationData) {
            summarizationData.messages_summarized = event.messages_summarized;
            summarizationData.context_usage_after = event.context_usage_after;
            summarizationData.summary = event.summary;
            
            content = {
              type: 'summarization',
              title: `✅ Summarized ${event.messages_summarized} messages. Context usage: ${summarizationData.context_usage_before.toFixed(1)}% → ${event.context_usage_after.toFixed(1)}%`,
              summary: event.summary,
              messages_summarized: event.messages_summarized,
              context_usage_before: summarizationData.context_usage_before,
              context_usage_after: event.context_usage_after,
            } as SummarizationContent;
          }
          isComplete = false;
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
                  role: role,
                  content: contentToFinalize,
                  tool_calls: role === 'assistant' ? toolCallsToFinalize : []
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

    // Update the current response state
    if (content || toolCalls.size > 0) {
      setCurrentResponse({
        role: role,
        content: content,
        toolCalls: toolCalls,
        summarizationData: summarizationData
      });
    } else {
      setCurrentResponse(null);
    }

    setIsStreamActive(!isComplete); // If we have a complete response, stop streaming
  }, [events]);

  // Only reconstruct messages array when baseMessages or currentResponse actually change
  const messages = useMemo(() => {
    if (!currentResponse || (currentResponse.content === '' && currentResponse.toolCalls.size === 0)) {
      return baseMessages;
    }
    
    return [
      ...baseMessages,
      {
        role: currentResponse.role,
        content: currentResponse.content,
        tool_calls: currentResponse.role === 'assistant' ? Array.from(currentResponse.toolCalls.values()) : []
      } as Message
    ];
  }, [baseMessages, currentResponse]);

  const addUserMessage = useCallback((content: string) => {
    const userMessage: UserMessage = {
      role: 'user',
      content
    };
    setBaseMessages(prev => [...prev, userMessage]);
    setCurrentResponse(null);
    setIsStreamActive(true); // Indicate that a new user message has been added
  }, []);

  const loadConversation = useCallback((conversationMessages: Message[]) => {
    setBaseMessages(conversationMessages);
    setCurrentResponse(null);
    setIsStreamActive(false); // Reset stream state when loading a conversation
    lastProcessedEventId.current = null; // Reset event processing state
  }, []);

  const clearConversation = useCallback(() => {
    setBaseMessages([]);
    setCurrentResponse(null);
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