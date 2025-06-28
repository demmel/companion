import { Send } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  onClear?: () => void;
  clearDisabled?: boolean;
  itemCount?: number;
  scrollMode?: string;
}

export function ChatInput({ 
  value, 
  onChange, 
  onSubmit, 
  disabled = false,
  placeholder = "Type your message...",
  onClear,
  clearDisabled = false,
  itemCount = 0,
  scrollMode = "Auto-scroll"
}: ChatInputProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!value.trim() || disabled) return;
    
    onSubmit(value);
  };

  return (
    <div className="bg-white border-t px-4 py-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500"
        />
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          <Send size={16} />
        </button>
      </form>
      
      <div className="flex justify-between items-center mt-2">
        {onClear && (
          <button
            onClick={onClear}
            disabled={clearDisabled}
            className="text-xs text-gray-500 hover:text-gray-700 underline disabled:text-gray-300 disabled:cursor-not-allowed disabled:no-underline"
          >
            Clear conversation
          </button>
        )}
        <span className="text-xs text-gray-400">
          {itemCount} items • {scrollMode}
        </span>
      </div>
    </div>
  );
}