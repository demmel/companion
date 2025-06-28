import { Send } from 'lucide-react';
import { css } from '@styled-system/css';

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
    <div className={css({ 
      bg: 'gray.800', 
      borderTop: '1px solid', 
      borderColor: 'gray.700', 
      p: 4 
    })}>
      <form onSubmit={handleSubmit} className={css({ 
        display: 'flex', 
        gap: 3 
      })}>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={css({ 
            flex: 1, 
            px: 4, 
            py: 3, 
            bg: 'gray.700', 
            border: '1px solid', 
            borderColor: 'gray.600', 
            rounded: 'lg', 
            color: 'white', 
            _placeholder: { color: 'gray.400' },
            _focus: { 
              outline: 'none', 
              borderColor: 'blue.500' 
            },
            _disabled: { 
              bg: 'gray.800', 
              color: 'gray.500' 
            }
          })}
        />
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className={css({ 
            px: 4, 
            py: 3, 
            bg: 'blue.600', 
            color: 'white', 
            rounded: 'lg', 
            _hover: { bg: 'blue.700' },
            _focus: { outline: 'none' },
            _disabled: { 
              bg: 'gray.600', 
              cursor: 'not-allowed' 
            },
            transition: 'colors'
          })}
        >
          <Send size={16} />
        </button>
      </form>
      
      <div className={css({ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mt: 3 
      })}>
        {onClear && (
          <button
            onClick={onClear}
            disabled={clearDisabled}
            className={css({ 
              fontSize: 'xs', 
              color: 'gray.400', 
              _hover: { color: 'red.400' },
              transition: 'colors',
              _disabled: { 
                color: 'gray.600', 
                cursor: 'not-allowed' 
              }
            })}
          >
            Clear conversation
          </button>
        )}
        <span className={css({ 
          fontSize: 'xs', 
          color: 'gray.500' 
        })}>
          {itemCount} items • {scrollMode}
        </span>
      </div>
    </div>
  );
}