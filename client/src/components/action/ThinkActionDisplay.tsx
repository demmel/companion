import { useState } from "react";
import { css } from "@styled-system/css";
import { Loader2, ChevronDown } from "lucide-react";
import { ThinkAction } from "@/types";
import { useActionAudio } from "@/hooks/useActionAudio";
import { PlayButton } from "./PlayButton";

interface ThinkActionDisplayProps {
  action: ThinkAction;
  triggerId: string;
  actionIndex: number;
}

export function ThinkActionDisplay({
  action,
  triggerId,
  actionIndex,
}: ThinkActionDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isStreaming = action.status.type === "streaming";
  const result =
    action.status.type === "error"
      ? `Error: ${action.status.error}`
      : action.status.result;
  const hasContent = result?.trim().length > 0;
  const label = isStreaming ? "Thinking..." : "Thoughts";

  const { playState, handlePlayClick } = useActionAudio({
    triggerId,
    actionIndex,
  });

  // Don't show play button while streaming
  const showPlayButton = action.status.type === "success";

  return (
    <div
      className={css({
        border: "1px solid",
        borderRadius: "md",
        borderColor: "gray.600",
        bg: "gray.800",
      })}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={!hasContent && !isStreaming}
        className={css({
          w: "full",
          px: 3,
          py: 2,
          textAlign: "left",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: "xl",
          fontWeight: "medium",
          color: "gray.400",
          _hover:
            hasContent || isStreaming
              ? {
                  bg: "gray.700",
                }
              : {},
          cursor: hasContent || isStreaming ? "pointer" : "default",
          _disabled: { cursor: "default" },
        })}
      >
        <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
          {isStreaming ? (
            <Loader2
              size={14}
              className={css({ animation: "spin 1s linear infinite" })}
            />
          ) : (
            <span className={css({ fontSize: "sm", opacity: 0.7 })}>💭</span>
          )}
          <span>{label}</span>
        </div>
        {(hasContent || isStreaming) && (
          <ChevronDown
            size={14}
            className={css({
              transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
            })}
          />
        )}
      </button>

      {isExpanded && (hasContent || isStreaming) && (
        <div
          className={css({
            px: 3,
            pb: 3,
            borderTop: "1px solid",
            borderColor: "gray.700",
          })}
        >
          {/* Play button */}
          {showPlayButton && (
            <div className={css({ pt: 2, pb: 2 })}>
              <PlayButton playState={playState} onClick={handlePlayClick} />
            </div>
          )}

          {/* Thoughts content */}
          <div
            className={css({
              fontSize: "xl",
              whiteSpace: "pre-wrap",
              p: 2,
              borderRadius: "sm",
              border: "1px solid",
              color: "gray.300",
              bg: "gray.900",
              borderColor: "gray.700",
            })}
          >
            {result?.trim() || (isStreaming ? "..." : "No thoughts recorded")}
            {isStreaming && result && (
              <span
                className={css({
                  animation: "blink 1s infinite",
                  color: "gray.500",
                })}
              >
                ▍
              </span>
            )}
          </div>
        </div>
      )}

      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}
