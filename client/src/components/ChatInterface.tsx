import { useState, useCallback, useEffect, useMemo } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { useStreamBatcher } from '../hooks/useStreamBatcher';
import { useConversation } from '../hooks/useConversation';
import { useSmartScroll } from '../hooks/useSmartScroll';
import { ChatHeader } from './ChatHeader';
import { ChatInput } from './ChatInput';
import { AgentEvent } from '../types';
import { AgentClient } from '../client';
import { getPresenterForConfig } from '../presenters';

interface ChatInterfaceProps {
  client: AgentClient;
}

export function ChatInterface({ client }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');
  const [configName, setConfigName] = useState<string>('general');
  const [agentState, setAgentState] = useState<Record<string, any>>({});
  
  // New architecture: batch events then convert to structured messages
  const { events, queueEvent, clearEvents } = useStreamBatcher(50);

  useMemo(() => {
    console.log('Current events:', events);
  }, [events]);

  const { messages, isStreamActive, addUserMessage, loadConversation, clearConversation } = useConversation(events);

  useMemo(() => {
    console.log('Current messages:', messages);
  }, [messages]);

  // Get the appropriate presenter component
  const PresenterComponent = getPresenterForConfig(configName);
  
  const handleMessage = useCallback((event: AgentEvent) => {
    queueEvent(event);
  }, [queueEvent]);

  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
  }, []);

  const { isConnected, isConnecting, sendMessage } = useWebSocket({
    url: client.chatWsUrl,
    onMessage: handleMessage,
    onError: handleError
  });

  const {
    isUserAtBottom,
    messagesEndRef,
    messagesContainerRef,
    handleScroll,
    setUserAtBottom
  } = useSmartScroll({ 
    items: messages 
  });

  const handleSubmit = (message: string) => {
    // Add user message to conversation
    addUserMessage(message);
    
    // Send to server
    sendMessage(message);
    setInputValue('');
    
    // When user sends a message, they probably want to see the response
    setUserAtBottom(true);
  };

  // Load initial data on mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Load config and state in parallel
        const [config, state, conversationData] = await Promise.all([
          client.getConfig(),
          client.getState(),
          client.getConversation()
        ]);
        
        setConfigName(config.name);
        setAgentState(state);
        
        if (conversationData.length > 0) {
          loadConversation(conversationData);
        }
      } catch (error) {
        console.error('Failed to load initial data:', error);
      }
    };

    loadInitialData();
  }, [client, loadConversation]);

  const handleClear = async () => {
    try {
      await client.reset();
    } catch (error) {
      console.error('Error resetting server:', error);
    }
    
    clearEvents();
    clearConversation();
    setUserAtBottom(true);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <ChatHeader 
        isConnected={isConnected}
        isConnecting={isConnecting}
      />

      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-4"
      >
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p>Start a conversation with the agent!</p>
            <p className="text-sm mt-2">Try: "Please roleplay as Elena, a mysterious vampire."</p>
          </div>
        )}
        
        {messages.length > 0 && (
          <PresenterComponent 
            messages={messages}
            isStreamActive={isStreamActive}
            agentState={agentState}
          />
        )}
      </div>
      
      {/* Scroll anchor */}
      <div ref={messagesEndRef} />

      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        disabled={!isConnected}
        onClear={handleClear}
        clearDisabled={isStreamActive}
        itemCount={messages.length}
        scrollMode={isUserAtBottom ? 'Auto-scroll' : 'Manual scroll'}
      />
    </div>
  );
}