import { forwardRef } from 'react';
import { MessageItem } from './MessageItem';
import { Message } from '../types';

interface ChatMessagesProps {
  messages: Message[];
  isStreamActive?: boolean;
  onScroll?: (e: React.UIEvent<HTMLDivElement>) => void;
  className?: string;
}

export const ChatMessages = forwardRef<HTMLDivElement, ChatMessagesProps>(
  function ChatMessages({ messages, isStreamActive = false, onScroll, className = '' }, ref) {
    return (
      <div 
        ref={ref}
        onScroll={onScroll}
        className={`flex-1 overflow-y-auto px-4 py-4 space-y-4 ${className}`}
      >
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p>Start a conversation with the agent!</p>
            <p className="text-sm mt-2">Try: "Please roleplay as Elena, a mysterious vampire."</p>
          </div>
        )}
        
        {messages.map((message, index) => (
          <div key={index} className="prose prose-sm max-w-none">
            <MessageItem message={message} />
          </div>
        ))}
        
        {/* Show streaming cursor when response is active */}
        {isStreamActive && messages.length > 0 && (
          <div className="prose prose-sm max-w-none">
            <span className="animate-pulse text-gray-500">▋</span>
          </div>
        )}
      </div>
    );
  }
);