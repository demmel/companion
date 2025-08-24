import { css } from "@styled-system/css";
import { AutoWakeupToggle } from "@/components/AutoWakeupToggle";
import { AgentClient } from "@/client";

interface ChatHeaderProps {
  title?: string;
  isConnected?: boolean;
  isConnecting?: boolean;
  client: AgentClient;
}

export function ChatHeader({ title = "Agent Chat", client }: ChatHeaderProps) {
  return (
    <div
      className={css({
        bg: "gray.800",
        borderBottom: "1px solid",
        borderColor: "gray.600",
        px: 4,
        py: 3,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      })}
    >
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          gap: 3,
        })}
      >
        <div
          className={css({
            w: 6,
            h: 6,
            rounded: "full",
            bg: "blue.600",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          })}
        >
          <span
            className={css({
              color: "white",
              fontSize: "xs",
            })}
          >
            🤖
          </span>
        </div>
        <h1
          className={css({
            fontSize: "lg",
            fontWeight: "medium",
            color: "white",
          })}
        >
          {title}
        </h1>
      </div>
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          gap: 3,
        })}
      >
        <AutoWakeupToggle client={client} />
      </div>
    </div>
  );
}
