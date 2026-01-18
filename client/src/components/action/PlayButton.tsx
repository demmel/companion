import { css } from "@styled-system/css";
import { Play, Square, Loader2 } from "lucide-react";
import { PlayState } from "@/hooks/useAudioPlayback";

interface PlayButtonProps {
  playState: PlayState;
  onClick: () => void;
}

export function PlayButton({ playState, onClick }: PlayButtonProps) {
  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 2,
      })}
    >
      <button
        onClick={onClick}
        className={css({
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          w: 8,
          h: 8,
          borderRadius: "full",
          bg: "gray.700",
          color: "gray.200",
          border: "none",
          cursor: "pointer",
          transition: "all 0.15s ease",
          _hover: {
            bg: "gray.600",
            color: "white",
          },
          _active: {
            transform: "scale(0.95)",
          },
        })}
        title={
          playState === "idle"
            ? "Play audio"
            : playState === "loading"
              ? "Loading..."
              : "Stop audio"
        }
      >
        {playState === "loading" && (
          <Loader2
            size={16}
            className={css({ animation: "spin 1s linear infinite" })}
          />
        )}
        {playState === "playing" && <Square size={14} />}
        {playState === "idle" && <Play size={16} />}
      </button>
      <span
        className={css({
          fontSize: "xs",
          color: "gray.500",
        })}
      >
        {playState === "loading"
          ? "Generating audio..."
          : playState === "playing"
            ? "Playing"
            : "Listen"}
      </span>
    </div>
  );
}
