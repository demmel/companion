import { css } from "@styled-system/css";
import { Loader2, Heart } from "lucide-react";
import { UpdateMoodAction } from "@/types";

interface UpdateMoodActionDisplayProps {
  action: UpdateMoodAction;
}

export function UpdateMoodActionDisplay({
  action,
}: UpdateMoodActionDisplayProps) {
  const isStreaming = action.status.type === "streaming";
  const result =
    action.status.type === "error"
      ? `Error: ${action.status.error}`
      : action.status.result;
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
            {action.context_given || "Adjusting mood..."}
          </span>
        ) : (
          <span>{result}</span>
        )}
      </div>
    </div>
  );
}
