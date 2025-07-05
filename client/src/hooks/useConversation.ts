import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { AgentEvent, Message, UserMessage, UserContent, AgentContent, SystemContent, SummarizationContent, ToolCall, ToolCallFinished, SystemMessage } from '../types';
import { debug } from '@/utils/debug';

// Builder types - same structure as final messages but content can be empty during building
interface UserMessageBuilder {
  role: 'user';
  content: UserContent; // Required but can be empty array []
}

interface AgentMessageBuilder {
  role: 'assistant';
  content: AgentContent; // Required but can be empty array []
  tool_calls: ToolCall[]; // Always present, can be empty array
}

interface SystemMessageBuilder {
  role: 'system';
  content: SystemContent; // Required but can be empty array []
  // No tool_calls field - system messages don't have them
  summarizationStartData?: {
    context_usage_before: number;
  };
}

type MessageBuilder = UserMessageBuilder | AgentMessageBuilder | SystemMessageBuilder;

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
  const [currentResponse, setCurrentResponse] = useState<MessageBuilder | null>(null);
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

    let response: MessageBuilder | null = currentResponse;
    let isComplete = false;

    // Helper to safely finalize a response and avoid closure bugs
    const finalizeResponse = (response: MessageBuilder) => {
      setBaseMessages(prev => [...prev, response as Message]);
    };

    for (const event of events) {
      if (lastProcessedEventId.current !== null && event.id <= lastProcessedEventId.current) {
        // Skip already processed events
        continue;
      }

      lastProcessedEventId.current = event.id;

      debug.log('Processing event:', event);

      switch (event.type) {
        case 'text':
          if (response && (response.role !== 'assistant' || response.tool_calls.length > 0)) {
            finalizeResponse(response);
            response = null;
          }

          if (!response) {
            response = {
              role: 'assistant',
              content: [],
              tool_calls: []
            };
          }

          // Add text content (we know response.role === 'assistant')
          const lastContent = response.content[response.content.length - 1];
          if (lastContent && lastContent.type === 'text') {
            // Append to existing text content
            lastContent.text += event.content;
          } else {
            // Create new text content
            response.content.push({ type: 'text', text: event.content });
          }

          isComplete = false;
          break;
        case 'tool_started':
          // Tool events are always assistant responses
          // If we have a system response, finalize it first
          if (response && response.role !== 'assistant') {
            finalizeResponse(response);
            response = null;
          }

          if (!response) {
            response = {
              role: 'assistant',
              content: [],
              tool_calls: []
            };
          }

          // Add tool call (we know response.role === 'assistant')
          response.tool_calls.push({
            tool_id: event.tool_id,
            tool_name: event.tool_name,
            type: 'started',
            parameters: event.parameters,
          });

          isComplete = false;
          break;
        case 'tool_finished':
          // Find by tool_id and replace with finished version
          if (!response || response.role !== 'assistant') {
            console.warn('Received tool_finished event but current response is not an assistant message. Ignoring.');
            continue; // Ignore if current response is not an assistant message
          }

          const toolIndex = response.tool_calls.findIndex(t => t.tool_id === event.tool_id);
          if (toolIndex === -1) {
            console.warn(`Tool finished event for unknown tool_id: ${event.tool_id}. Ignoring.`);
            continue; // Ignore if tool_id not found
          }

          const tool = response.tool_calls[toolIndex];
          response.tool_calls[toolIndex] = {
            ...tool,
            type: 'finished',
            result: {
              type: event.result_type,
              content: event.result
            }
          } as ToolCallFinished;

          isComplete = false;
          break;
        case 'summarization_started':
          // Summarization events are always system responses
          // Finalize any existing response first
          if (response) {
            finalizeResponse(response);
            response = null;
          }

          response = {
            role: 'system',
            content: [{
              type: 'text',
              text: `📝 Summarizing ${event.messages_to_summarize} older messages to manage context (${event.context_usage_before.toFixed(1)}% usage)...`
            }],
            summarizationStartData: {
              context_usage_before: event.context_usage_before
            }
          };

          isComplete = false;
          break;
        case 'summarization_finished':
          // Replace the system response content with structured summarization
          if (!response || response.role !== 'system') {
            console.warn('Received summarization_finished event but current response is not a system message. Ignoring.');
            continue; // Ignore if current response is not a system message
          }

          const summarizationStartData = response.summarizationStartData;
          if (!summarizationStartData) {
            console.warn('Received summarization_finished event but no summarization start data found. Ignoring.');
            continue; // Ignore if no summarization start data
          }

          const summaryContent = {
            type: 'summarization',
            title: `✅ Summarized ${event.messages_summarized} messages. Context usage: ${summarizationStartData.context_usage_before.toFixed(1)}% → ${event.context_usage_after.toFixed(1)}%`,
            summary: event.summary,
            messages_summarized: event.messages_summarized,
            context_usage_before: summarizationStartData.context_usage_before,
            context_usage_after: event.context_usage_after,
          } as SummarizationContent;

          // Remove the summarization data before finalizing (it's not part of the final message)
          const finalResponse = {
            role: response.role,
            content: [summaryContent]
          } as SystemMessage;

          // Finalize the system message immediately
          setBaseMessages(prev => [...prev, finalResponse]);
          response = null;

          isComplete = false; // More events may follow
          break;
        case 'response_complete':
          // Finalize any remaining response
          if (response) {
            finalizeResponse(response);
            response = null;
          }

          isComplete = true;
      }

      debug.log('Updated response:', response);
    }

    // Update the current response state
    setCurrentResponse(response);
    setIsStreamActive(!isComplete); // If we have a complete response, stop streaming
  }, [events]);

  // Only reconstruct messages array when baseMessages or currentResponse actually change
  const messages = useMemo(() => {
    if (!currentResponse || (currentResponse.content.length === 0 &&
      (currentResponse.role !== 'assistant' || currentResponse.tool_calls.length === 0))) {
      return baseMessages;
    }

    // currentResponse is already a valid Message, just cast it
    return [
      ...baseMessages,
      currentResponse as Message
    ];
  }, [baseMessages, currentResponse]);

  const addUserMessage = useCallback((content: string) => {
    const userMessage: UserMessage = {
      role: 'user',
      content: [{ type: 'text', text: content }]
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