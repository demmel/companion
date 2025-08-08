import { useState } from "react";
import { css } from "@styled-system/css";
import { ChevronDown } from "lucide-react";
import { SpeakAction } from "@/types";

interface SpeakActionDisplayProps {
  action: SpeakAction;
}

export function SpeakActionDisplay({ action }: SpeakActionDisplayProps) {
  const [showContext, setShowContext] = useState(false);
  
  const isStreaming = action.status.type === "streaming";
  const result = action.status.type === "error" ? 
    `Error: ${action.status.error}` : 
    action.status.result;
  
  return (
    <div>
      {/* Main speech content - always visible */}
      <div
        className={css({
          p: 3,
          fontSize: "xl",
          lineHeight: "relaxed",
          color: action.status.type === "error" ? "red.300" : "gray.200",
          whiteSpace: "pre-wrap",
        })}
      >
        {result}
        {isStreaming && (
          <span className={css({ 
            animation: "blink 1s infinite",
            color: "gray.500" 
          })}>▍</span>
        )}
      </div>

      {/* Optional context section for debugging */}
      {action.context_given && (
        <div className={css({ borderTop: "1px solid", borderColor: "gray.700" })}>
          <button
            onClick={() => setShowContext(!showContext)}
            className={css({
              w: "full",
              px: 3,
              py: 1,
              textAlign: "left",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "sm",
              color: "gray.500",
              _hover: { bg: "gray.800" },
            })}
          >
            <span>Why this response</span>
            <ChevronDown 
              size={12}
              className={css({
                transform: showContext ? "rotate(180deg)" : "rotate(0deg)",
                transition: "transform 0.2s ease",
              })}
            />
          </button>
          
          {showContext && (
            <div className={css({ 
              px: 3, 
              pb: 2, 
              fontSize: "sm", 
              color: "gray.400",
              fontStyle: "italic" 
            })}>
              {action.context_given}
            </div>
          )}
        </div>
      )}

      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
        `}
      </style>
    </div>
  );
}