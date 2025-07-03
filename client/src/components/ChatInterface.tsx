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
import { debug } from '@/utils/debug';

interface ChatInterfaceProps {
  client: AgentClient;
}

interface ContextInfo {
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  conversation_messages: number;
  approaching_limit: boolean;
}

export function ChatInterface({ client }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');
  const [configName, setConfigName] = useState<string>('general');
  const [agentState, setAgentState] = useState<Record<string, any>>({});
  const [contextInfo, setContextInfo] = useState<ContextInfo | null>(null);
  
  // New architecture: batch events then convert to structured messages
  const { events, queueEvent, clearEvents } = useStreamBatcher(50);

  useMemo(() => {
    debug.log('Events:', events);
  }, [events]);

  const { messages, isStreamActive, addUserMessage, loadConversation, clearConversation } = useConversation(events);

  useEffect(() => {
    debug.log('Messages:', messages);
  }, [messages]);

  // Get the appropriate presenter component
  const PresenterComponent = getPresenterForConfig(configName);
  
  const handleMessage = useCallback((event: AgentEvent) => {
    // Handle response_complete events to update context info
    if (event.type === 'response_complete') {
      setContextInfo({
        estimated_tokens: (event as any).estimated_tokens,
        context_limit: (event as any).context_limit,
        usage_percentage: (event as any).usage_percentage,
        conversation_messages: (event as any).conversation_messages,
        approaching_limit: (event as any).approaching_limit,
      });
    }
    
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
        // Load config, state, conversation, and context info in parallel
        const [config, state, conversationData, contextData] = await Promise.all([
          client.getConfig(),
          client.getState(),
          client.getConversation(),
          client.getContextInfo()
        ]);
        
        setConfigName(config.name);
        setAgentState(state);
        setContextInfo({
          estimated_tokens: contextData.estimated_tokens,
          context_limit: contextData.context_limit,
          usage_percentage: contextData.usage_percentage,
          conversation_messages: contextData.conversation_messages,
          approaching_limit: contextData.approaching_limit,
        });
        
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
    setContextInfo(null);
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
                fontSize: 'xl', 
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
                  fontSize: 'xl' 
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
        disabled={!isConnected || isConnecting || isStreamActive}
        onClear={handleClear}
        clearDisabled={isStreamActive}
        contextInfo={contextInfo}
      />
    </div>
  );
}