import { Send } from "lucide-react";
import { css } from "@styled-system/css";
import { useRef, useEffect } from "react";

interface ContextInfo {
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  conversation_messages: number;
  approaching_limit: boolean;
}

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  onClear?: () => void;
  clearDisabled?: boolean;
  contextInfo?: ContextInfo | null;
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = "Type your message...",
  onClear,
  clearDisabled = false,
  contextInfo,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const resetTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = "56px";
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!value.trim() || disabled) return;

    onSubmit(value);
    resetTextareaHeight();
  };

  // Reset height when value is cleared externally
  useEffect(() => {
    if (!value && textareaRef.current) {
      resetTextareaHeight();
    }
  }, [value]);

  return (
    <div
      className={css({
        bg: "gray.800",
        borderTop: "1px solid",
        borderColor: "gray.700",
        p: 4,
      })}
    >
      <form
        onSubmit={handleSubmit}
        className={css({
          display: "flex",
          gap: 3,
          alignItems: "flex-end",
        })}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!value.trim() || disabled) return;
              onSubmit(value);
              resetTextareaHeight();
            }
          }}
          className={css({
            flex: 1,
            px: 4,
            py: 3,
            bg: "gray.700",
            border: "1px solid",
            borderColor: "gray.600",
            rounded: "lg",
            color: "white",
            fontSize: "xl",
            resize: "none",
            minHeight: "56px",
            maxHeight: "120px",
            lineHeight: "1.5",
            _placeholder: { color: "gray.400" },
            _focus: {
              outline: "none",
              borderColor: "blue.500",
            },
            _disabled: {
              bg: "gray.800",
              color: "gray.500",
            },
          })}
          style={{
            height: "auto",
            minHeight: "56px",
          }}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement;
            target.style.height = "auto";
            target.style.height = Math.min(target.scrollHeight, 120) + "px";
          }}
        />
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className={css({
            px: 4,
            py: 3,
            bg: "blue.600",
            color: "white",
            rounded: "lg",
            minHeight: "56px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            _hover: { bg: "blue.700" },
            _focus: { outline: "none" },
            _disabled: {
              bg: "gray.600",
              cursor: "not-allowed",
            },
            transition: "colors",
          })}
        >
          <Send size={16} />
        </button>
      </form>

      <div
        className={css({
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mt: 3,
        })}
      >
        {onClear && (
          <button
            onClick={onClear}
            disabled={clearDisabled}
            className={css({
              fontSize: "sm",
              color: "gray.400",
              _hover: { color: "red.400" },
              transition: "colors",
              _disabled: {
                color: "gray.600",
                cursor: "not-allowed",
              },
            })}
          >
            Clear
          </button>
        )}
        <span
          className={css({
            fontSize: "sm",
            color: contextInfo?.approaching_limit ? "yellow.400" : "gray.500",
          })}
        >
          {contextInfo
            ? `${Math.round((contextInfo.estimated_tokens / 1000) * 10) / 10}k/${Math.round(contextInfo.context_limit / 1000)}k tokens (${Math.round(contextInfo.usage_percentage)}%) • ${contextInfo.conversation_messages} messages`
            : "Loading..."}
        </span>
      </div>
    </div>
  );
}
