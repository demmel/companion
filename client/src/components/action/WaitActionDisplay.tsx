import { css } from "@styled-system/css";
import { Pause } from "lucide-react";
import { WaitAction } from "@/types";

interface WaitActionDisplayProps {
  action: WaitAction;
}

export function WaitActionDisplay({}: WaitActionDisplayProps) {
  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 3,
        p: 2,
        bg: "gray.800",
        border: "1px solid",
        borderColor: "gray.700",
        rounded: "md",
        fontSize: "md",
        color: "gray.400",
      })}
    >
      <Pause size={16} className={css({ color: "gray.400" })} />

      <div className={css({ flex: 1, fontStyle: "italic" })}>
        Waiting for something to happen...
      </div>
    </div>
  );
}
