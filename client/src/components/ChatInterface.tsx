import { useState, useCallback, useEffect, useMemo } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useStreamBatcher } from '@/hooks/useStreamBatcher';
import { useConversation } from '@/hooks/useConversation';
import { useSmartScroll } from '@/hooks/useSmartScroll';
import { ChatHeader } from '@/components/ChatHeader';
import { ChatInput } from '@/components/ChatInput';
import { AgentEvent } from '@/types';
import { AgentClient } from '@/client';
import { getPresenterForConfig } from '@/presenters';
import { css } from '@styled-system/css';

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
    <div className={css({ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh', 
      bg: 'gray.900' 
    })}>
      <ChatHeader 
        isConnected={isConnected}
        isConnecting={isConnecting}
      />

      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className={css({ 
          flex: 1, 
          overflowY: 'auto', 
          px: 4, 
          py: 4 
        })}
      >
        {messages.length === 0 && (
          <div className={css({ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: 'full', 
            textAlign: 'center' 
          })}>
            <div className={css({ maxWidth: 'md' })}>
              <div className={css({ 
                fontSize: '4xl', 
                mb: 6, 
                opacity: 0.5 
              })}>💬</div>
              <h2 className={css({ 
                fontSize: 'xl', 
                fontWeight: 'medium', 
                color: 'gray.300', 
                mb: 2 
              })}>Start a conversation</h2>
              <p className={css({ 
                color: 'gray.500', 
                fontSize: 'sm', 
                mb: 6 
              })}>Send a message to begin chatting with the agent</p>
              <div className={css({ 
                bg: 'gray.800', 
                border: '1px solid', 
                borderColor: 'gray.700', 
                rounded: 'lg', 
                p: 4, 
                textAlign: 'left' 
              })}>
                <p className={css({ 
                  fontSize: 'xs', 
                  color: 'gray.400', 
                  textTransform: 'uppercase', 
                  letterSpacing: 'wide', 
                  mb: 2 
                })}>Example</p>
                <p className={css({ 
                  color: 'gray.300', 
                  fontSize: 'sm' 
                })}>"Please roleplay as Elena, a mysterious vampire."</p>
              </div>
            </div>
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