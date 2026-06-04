import { css } from "@styled-system/css";
import { Loader2, Heart } from "lucide-react";
import { UpdateMoodAction } from "@/types";
import { isStreamingResult, resultText } from "./actionResult";

interface UpdateMoodActionDisplayProps {
  action: UpdateMoodAction;
}

export function UpdateMoodActionDisplay({
  action,
}: UpdateMoodActionDisplayProps) {
  const isStreaming = isStreamingResult(action.result);
  const result = resultText(
    action.result,
    (content) =>
      `Mood changed from ${content.old_mood} (${content.old_intensity}) to ${content.new_mood} (${content.new_intensity}): ${content.reason}`,
  );
  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 3,
        p: 2,
        bg: "blue.900/20",
        border: "1px solid",
        borderColor: "blue.700",
        rounded: "md",
        fontSize: "xl",
      })}
    >
      {isStreaming ? (
        <Loader2
          size={16}
          className={css({
            animation: "spin 1s linear infinite",
            color: "blue.500",
          })}
        />
      ) : (
        <Heart size={16} className={css({ color: "blue.500" })} />
      )}

      <div className={css({ flex: 1, color: "blue.300" })}>
        {isStreaming ? (
          <span className={css({ fontStyle: "italic" })}>
            Adjusting mood...
          </span>
        ) : (
          <span>{result}</span>
        )}
      </div>
    </div>
  );
}
