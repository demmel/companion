import { css } from "@styled-system/css";
import { Loader2, Plus } from "lucide-react";
import { AddPriorityAction } from "@/types";
import { isStreamingResult, resultText } from "./actionResult";

interface AddPriorityActionDisplayProps {
  action: AddPriorityAction;
}

export function AddPriorityActionDisplay({
  action,
}: AddPriorityActionDisplayProps) {
  const isStreaming = isStreamingResult(action.result);
  const result = resultText(action.result, (content) =>
    content.result.type === "success"
      ? `Added new priority: "${content.content}"`
      : `Priority "${content.content}" is similar to an existing priority`,
  );

  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 3,
        p: 2,
        bg: "green.900/20",
        border: "1px solid",
        borderColor: "green.700",
        rounded: "md",
        fontSize: "sm",
      })}
    >
      {isStreaming ? (
        <Loader2
          size={16}
          className={css({
            animation: "spin 1s linear infinite",
            color: "green.500",
          })}
        />
      ) : (
        <Plus size={16} className={css({ color: "green.500" })} />
      )}

      <div className={css({ flex: 1, color: "green.300" })}>
        {isStreaming ? (
          <span className={css({ fontStyle: "italic" })}>
            Adding priority...
          </span>
        ) : (
          <span>{result}</span>
        )}
      </div>
    </div>
  );
}
