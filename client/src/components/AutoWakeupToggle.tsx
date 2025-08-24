import { useState, useEffect } from "react";
import { Clock } from "lucide-react";
import { css } from "@styled-system/css";
import { AgentClient } from "@/client";

interface AutoWakeupToggleProps {
  client: AgentClient;
}

export function AutoWakeupToggle({ client }: AutoWakeupToggleProps) {
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [delayMinutes, setDelayMinutes] = useState(5);

  // Load initial status
  useEffect(() => {
    const loadStatus = async () => {
      try {
        const status = await client.getAutoWakeupStatus();
        setEnabled(status.enabled);
        setDelayMinutes(Math.round(status.delay_seconds / 60));
      } catch (error) {
        console.error("Failed to load auto-wakeup status:", error);
      }
    };

    loadStatus();
  }, [client]);

  const handleToggle = async () => {
    if (loading) return;

    setLoading(true);
    try {
      const response = await client.setAutoWakeupEnabled(!enabled);
      setEnabled(response.enabled);
    } catch (error) {
      console.error("Failed to toggle auto-wakeup:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={css({
        display: "flex",
        alignItems: "center",
        gap: 2,
      })}
    >
      <Clock
        size={16}
        className={css({
          color: enabled ? "blue.400" : "gray.500",
          transition: "colors",
        })}
      />
      <button
        onClick={handleToggle}
        disabled={loading}
        title={`Auto-wakeup: ${enabled ? "Enabled" : "Disabled"} (${delayMinutes}min)`}
        className={css({
          position: "relative",
          w: 10,
          h: 6,
          bg: enabled ? "blue.600" : "gray.600",
          rounded: "full",
          transition: "all 0.2s",
          cursor: loading ? "not-allowed" : "pointer",
          opacity: loading ? 0.5 : 1,
          _hover: {
            bg: loading ? (enabled ? "blue.600" : "gray.600") : (enabled ? "blue.700" : "gray.500"),
          },
        })}
      >
        <div
          className={css({
            position: "absolute",
            top: "1px",
            left: enabled ? "calc(100% - 21px)" : "1px",
            w: 5,
            h: 5,
            bg: "white",
            rounded: "full",
            transition: "all 0.2s",
            transform: "translateX(0)",
          })}
        />
      </button>
      <span
        className={css({
          fontSize: "sm",
          color: enabled ? "blue.400" : "gray.500",
          fontWeight: "medium",
          transition: "colors",
        })}
      >
        {delayMinutes}m
      </span>
    </div>
  );
}