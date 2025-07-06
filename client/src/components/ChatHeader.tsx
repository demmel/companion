import { Wifi, WifiOff } from "lucide-react";
import { css } from "@styled-system/css";

interface ChatHeaderProps {
  title?: string;
  isConnected?: boolean;
  isConnecting?: boolean;
}

export function ChatHeader({
  title = "Agent Chat",
  isConnected = false,
  isConnecting = false,
}: ChatHeaderProps) {
  const getConnectionStatus = () => {
    if (isConnecting)
      return { icon: Wifi, text: "Connecting...", color: "text-yellow-500" };
    if (isConnected)
      return { icon: Wifi, text: "Connected", color: "text-green-500" };
    return { icon: WifiOff, text: "Disconnected", color: "text-red-500" };
  };

  const status = getConnectionStatus();
  const StatusIcon = status.icon;

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
          gap: 2,
        })}
      >
        <StatusIcon size={14} className={status.color} />
        <span className={`${css({ fontSize: "xs" })} ${status.color}`}>
          {status.text}
        </span>
      </div>
    </div>
  );
}
