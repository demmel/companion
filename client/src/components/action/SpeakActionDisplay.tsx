import { css } from "@styled-system/css";
import { SpeakAction } from "@/types";
import { useActionAudio } from "@/hooks/useActionAudio";
import { PlayButton } from "./PlayButton";

interface SpeakActionDisplayProps {
  action: SpeakAction;
  triggerId: string;
  actionIndex: number;
}

export function SpeakActionDisplay({
  action,
  triggerId,
  actionIndex,
}: SpeakActionDisplayProps) {
  const isStreaming = action.status.type === "streaming";
  const result =
    action.status.type === "error"
      ? `Error: ${action.status.error}`
      : action.status.result;

  const { playState, handlePlayClick } = useActionAudio({
    triggerId,
    actionIndex,
  });

  // Don't show play button while streaming
  const showPlayButton = action.status.type === "success";

  return (
    <div>
      {/* Play button row */}
      {showPlayButton && (
        <div className={css({ px: 3, pt: 2 })}>
          <PlayButton playState={playState} onClick={handlePlayClick} />
        </div>
      )}

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
