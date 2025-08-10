import { css } from "@styled-system/css";
import { Loader2, Minus } from "lucide-react";
import { RemovePriorityAction } from "@/types";

interface RemovePriorityActionDisplayProps {
  action: RemovePriorityAction;
}

export function RemovePriorityActionDisplay({ action }: RemovePriorityActionDisplayProps) {
  const isStreaming = action.status.type === "streaming";
  const result = action.status.type === "error" ? 
    `Error: ${action.status.error}` : 
    action.status.result;
    
  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 3,
        p: 2,
        bg: "red.900/20",
        border: "1px solid",
        borderColor: "red.700",
        rounded: "md",
        fontSize: "sm",
      })}
    >
      {isStreaming ? (
        <Loader2 size={16} className={css({ animation: "spin 1s linear infinite", color: "red.500" })} />
      ) : (
        <Minus size={16} className={css({ color: "red.500" })} />
      )}
      
      <div className={css({ flex: 1, color: "red.300" })}>
        {isStreaming ? (
          <span className={css({ fontStyle: "italic" })}>
            {action.context_given || "Removing priority..."}
          </span>
        ) : (
          <span>{result}</span>
        )}
      </div>
    </div>
  );
}