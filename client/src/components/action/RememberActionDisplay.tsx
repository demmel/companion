import { useState } from "react";
import { css } from "@styled-system/css";
import { Loader2, Brain, ChevronDown } from "lucide-react";
import { RememberAction } from "@/types";
import { isStreamingResult } from "./actionResult";

interface RememberActionDisplayProps {
  action: RememberAction;
}

export function RememberActionDisplay({ action }: RememberActionDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isStreaming = isStreamingResult(action.result);
  const memories =
    action.result.type === "success" ? action.result.content.memories : [];
  const queries = "input" in action ? action.input.queries : [];
  const hasContent = memories.length > 0;

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
      {/* Header */}
      <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
        {isStreaming ? (
          <Loader2
            size={16}
            className={css({
              animation: "spin 1s linear infinite",
              color: "purple.400",
            })}
          />
        ) : (
          <Brain size={16} className={css({ color: "purple.400" })} />
        )}
        <div className={css({ flex: 1 })}>
          <div className={css({ color: "purple.300", fontWeight: "medium" })}>
            {isStreaming
              ? "Recalling memories..."
              : action.result.type === "failure"
                ? "Recall failed"
                : memories.length > 0
                  ? `Recalled ${memories.length} ${memories.length === 1 ? "memory" : "memories"}`
                  : "No memories found"}
          </div>
          {"input" in action && action.input.reason && (
            <div
              className={css({
                color: "purple.400",
                fontSize: "xs",
                fontStyle: "italic",
              })}
            >
              {action.input.reason}
            </div>
          )}
        </div>
      </div>

      {/* Queries that were run */}
      {queries.length > 0 && (
        <div className={css({ display: "flex", flexWrap: "wrap", gap: 1 })}>
          {queries.map((query, index) => (
            <span
              key={index}
              title={`${query.query_type} · importance ${query.importance}`}
              className={css({
                color: "purple.200",
                fontSize: "xs",
                px: 2,
                py: 0.5,
                bg: "purple.900/40",
                border: "1px solid",
                borderColor: "purple.800",
                rounded: "full",
              })}
            >
              {query.query_text}
            </span>
          ))}
        </div>
      )}

      {action.result.type === "failure" && (
        <div className={css({ color: "red.300", fontSize: "xs" })}>
          {action.result.error}
        </div>
      )}

      {/* Expandable retrieved memories */}
      {hasContent && (
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
              _hover: { bg: "purple.800/30" },
              cursor: "pointer",
            })}
          >
            <span>Recalled memories</span>
            <ChevronDown
              size={14}
              className={css({
                transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                transition: "transform 0.2s ease",
              })}
            />
          </button>

          {isExpanded && (
            <div
              className={css({
                display: "flex",
                flexDirection: "column",
                gap: 2,
                mx: 2,
                mb: 2,
              })}
            >
              {memories.map((memory) => (
                <div
                  key={memory.memory_id}
                  className={css({
                    p: 2,
                    bg: "purple.950/50",
                    border: "1px solid",
                    borderColor: "purple.800",
                    rounded: "sm",
                  })}
                >
                  <div
                    className={css({
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 2,
                      mb: 1,
                    })}
                  >
                    <span
                      className={css({ color: "purple.400", fontSize: "xs" })}
                    >
                      {new Date(memory.timestamp).toLocaleString()}
                    </span>
                    <span
                      className={css({
                        color: "purple.300",
                        fontSize: "xs",
                        px: 1.5,
                        py: 0.5,
                        bg: "purple.900/50",
                        rounded: "sm",
                      })}
                    >
                      {memory.confidence}
                    </span>
                  </div>
                  <div
                    className={css({
                      color: "purple.100",
                      fontSize: "sm",
                      lineHeight: "relaxed",
                      whiteSpace: "pre-wrap",
                    })}
                  >
                    {memory.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}
