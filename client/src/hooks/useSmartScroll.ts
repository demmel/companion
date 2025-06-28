import { useState, useEffect, useRef, useCallback } from 'react';

interface UseSmartScrollOptions {
  items: any[]; // Array of items that trigger scroll updates
  threshold?: number; // Distance from bottom to consider "at bottom" 
}

interface UseSmartScrollReturn {
  isUserAtBottom: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  messagesContainerRef: React.RefObject<HTMLDivElement | null>;
  handleScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  scrollToBottom: () => void;
  setUserAtBottom: (atBottom: boolean) => void;
}

export function useSmartScroll({ 
  items, 
  threshold = 100 
}: UseSmartScrollOptions): UseSmartScrollReturn {
  const [isUserAtBottom, setIsUserAtBottom] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, clientHeight, scrollHeight } = e.currentTarget;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - threshold;
    setIsUserAtBottom(isAtBottom);
  }, [threshold]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const setUserAtBottom = useCallback((atBottom: boolean) => {
    setIsUserAtBottom(atBottom);
  }, []);

  // Auto-scroll to bottom only if user is already at bottom
  useEffect(() => {
    if (isUserAtBottom) {
      scrollToBottom();
    }
  }, [items, isUserAtBottom, scrollToBottom]);

  return {
    isUserAtBottom,
    messagesEndRef,
    messagesContainerRef,
    handleScroll,
    scrollToBottom,
    setUserAtBottom
  };
}