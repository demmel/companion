import { useState, useCallback, useEffect, useMemo } from 'react';
import { AgentEvent, Message, UserMessage, AgentMessage, ToolCall } from '../types';

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
    isComplete: boolean;
  } | null>(null);

  // Process events into current agent response
  useEffect(() => {
    let agentContent = '';
    let toolCalls = new Map<string, ToolCall>();
    let isComplete = false;

    for (const event of events) {
      switch (event.type) {
        case 'text':
          // If we have finished tool calls, finalize that message first
          if (toolCalls.size > 0) {
            const finalToolMessage: AgentMessage = {
              role: 'assistant',
              content: agentContent,
              tool_calls: Array.from(toolCalls.values())
            };
            setBaseMessages(prev => [...prev, finalToolMessage]);
            
            // Reset for new text-only response
            agentContent = '';
            toolCalls = new Map();
          }

          agentContent += event.content;
          break;

        case 'tool_started':
          toolCalls.set(event.tool_id, {
            type: 'started',
            tool_name: event.tool_name,
            tool_id: event.tool_id,
            parameters: event.parameters
          });
          break;

        case 'tool_finished':
          const existingTool = toolCalls.get(event.tool_id);
          if (existingTool && existingTool.type === 'started') {
            toolCalls.set(event.tool_id, {
              type: 'finished',
              tool_name: existingTool.tool_name,
              tool_id: event.tool_id,
              parameters: existingTool.parameters,
              result: {
                type: event.result_type,
                content: event.result
              }
            });
          }
          break;

        case 'response_complete':
          isComplete = true;
          break;
      }
    }

    // Update current response (this makes it visible during streaming)
    if (agentContent || toolCalls.size > 0) {
      setCurrentAgentResponse({
        content: agentContent,
        toolCalls,
        isComplete
      });
    }

    // If stream is complete, move to base messages and clear current
    if (isComplete && (agentContent || toolCalls.size > 0)) {
      const finalMessage: AgentMessage = {
        role: 'assistant',
        content: agentContent,
        tool_calls: Array.from(toolCalls.values())
      };

      setBaseMessages(prev => [...prev, finalMessage]);
      setCurrentAgentResponse(null);
    }
  }, [events]);

  // Only reconstruct messages array when baseMessages or currentAgentResponse actually change
  const messages = useMemo(() => {
    if (!currentAgentResponse) {
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
  }, []);

  const loadConversation = useCallback((conversationMessages: Message[]) => {
    setBaseMessages(conversationMessages);
    setCurrentAgentResponse(null);
  }, []);

  const clearConversation = useCallback(() => {
    setBaseMessages([]);
    setCurrentAgentResponse(null);
  }, []);

  const isStreamActive = currentAgentResponse !== null && !currentAgentResponse.isComplete;

  return {
    messages,
    isStreamActive,
    addUserMessage,
    loadConversation,
    clearConversation
  };
}