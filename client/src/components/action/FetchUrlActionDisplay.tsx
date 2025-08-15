import { useState } from "react";
import { css } from "@styled-system/css";
import { Loader2, ExternalLink, ChevronDown } from "lucide-react";
import { FetchUrlAction } from "@/types";

interface FetchUrlActionDisplayProps {
  action: FetchUrlAction;
}

export function FetchUrlActionDisplay({ action }: FetchUrlActionDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const isStreaming = action.status.type === "streaming";
  const result = action.status.type === "error" ? 
    `Error: ${action.status.error}` : 
    action.status.result;
  const hasContent = result?.trim().length > 0;

  return (
    <div
      className={css({
        display: "flex",
        flexDirection: "column",
        gap: 2,
        p: 3,
        bg: "purple.900/20",
        border: "1px solid",
        borderColor: "purple.700",
        rounded: "md",
        fontSize: "sm",
      })}
    >
      {/* Header with icon and URL */}
      <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
        {isStreaming ? (
          <Loader2 size={16} className={css({ animation: "spin 1s linear infinite", color: "purple.500" })} />
        ) : (
          <ExternalLink size={16} className={css({ color: "purple.500" })} />
        )}
        
        <div className={css({ flex: 1 })}>
          <div className={css({ color: "purple.300", fontWeight: "medium" })}>
            {isStreaming ? "Fetching content..." : "Fetched URL"}
          </div>
          {action.url && (
            <div className={css({ color: "purple.200", fontSize: "xs", truncate: true })}>
              <a 
                href={action.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className={css({ 
                  color: "purple.200", 
                  textDecoration: "underline",
                  _hover: { color: "purple.100" }
                })}
              >
                {action.url}
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Looking for context */}
      {action.looking_for && (
        <div className={css({ color: "purple.400", fontSize: "xs", fontStyle: "italic" })}>
          Looking for: {action.looking_for}
        </div>
      )}

      {/* Expandable result content */}
      {(hasContent || isStreaming) && (
        <div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={css({
              w: "full",
              px: 2,
              py: 1,
              textAlign: "left",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "sm",
              color: "purple.400",
              _hover: {
                bg: "purple.800/30",
              },
              cursor: "pointer",
            })}
          >
            <span>{isStreaming ? "Processing..." : "Summary"}</span>
            <ChevronDown 
              size={14}
              className={css({
                transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                transition: "transform 0.2s ease",
              })}
            />
          </button>
          
          {isExpanded && (
            <div className={css({ 
              color: "purple.100", 
              fontSize: "xl",
              lineHeight: "relaxed",
              whiteSpace: "pre-wrap",
              p: 2,
              bg: "purple.950/50",
              rounded: "sm",
              border: "1px solid",
              borderColor: "purple.800",
              mx: 2,
              mb: 2
            })}>
              {result?.trim() || (isStreaming ? "..." : "No content available")}
              {isStreaming && result && (
                <span className={css({ 
                  animation: "blink 1s infinite",
                  color: "purple.500" 
                })}>▍</span>
              )}
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