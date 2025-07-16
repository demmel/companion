import { useState } from "react";
import { css } from "@styled-system/css";
import { ThoughtContent } from "@/types";

interface ThoughtBubbleProps {
  content: ThoughtContent;
  isStreaming?: boolean;
}

export function ThoughtBubble({ content, isStreaming = false }: ThoughtBubbleProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const label = isStreaming ? "Thinking..." : "Thoughts";
  const hasContent = content.text.trim().length > 0;

  return (
    <div
      className={css({
        border: "1px solid",
        borderColor: "gray.300",
        borderRadius: "md",
        mb: 2,
        bg: "gray.50",
        _dark: {
          borderColor: "gray.600",
          bg: "gray.800",
        },
      })}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={!hasContent}
        className={css({
          w: "full",
          px: 3,
          py: 2,
          textAlign: "left",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: "sm",
          fontWeight: "medium",
          color: "gray.600",
          _dark: { color: "gray.400" },
          _hover: hasContent ? { 
            bg: "gray.100",
            _dark: { bg: "gray.700" } 
          } : {},
          cursor: hasContent ? "pointer" : "default",
          _disabled: { cursor: "default" },
        })}
      >
        <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
          <span
            className={css({
              fontSize: "xs",
              opacity: isStreaming ? 1 : 0.7,
              ...(isStreaming && {
                animation: "shimmer 1.5s ease-in-out infinite",
              }),
            })}
          >
            💭
          </span>
          <span>{label}</span>
        </div>
        {hasContent && (
          <span
            className={css({
              fontSize: "xs",
              transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
            })}
          >
            ▼
          </span>
        )}
      </button>
      
      {isExpanded && hasContent && (
        <div
          className={css({
            px: 3,
            pb: 3,
            borderTop: "1px solid",
            borderColor: "gray.200",
            _dark: { borderColor: "gray.600" },
          })}
        >
          <div
            className={css({
              fontSize: "sm",
              color: "gray.700",
              whiteSpace: "pre-wrap",
              fontFamily: "mono",
              bg: "white",
              p: 2,
              borderRadius: "sm",
              border: "1px solid",
              borderColor: "gray.200",
              _dark: { 
                color: "gray.300",
                bg: "gray.900",
                borderColor: "gray.700",
              },
            })}
          >
            {content.text}
          </div>
        </div>
      )}
      
      <style>
        {`
          @keyframes shimmer {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
          }
        `}
      </style>
    </div>
  );
}