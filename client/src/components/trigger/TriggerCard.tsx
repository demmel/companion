import { css } from "@styled-system/css";
import {
  MessageCircle,
  User,
  Clock,
  Coffee,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { TriggerHistoryEntry } from "@/types";
import { ImageDisplay } from "../common/ImageDisplay";
import { useState } from "react";

interface TriggerCardProps {
  entry: TriggerHistoryEntry;
}

export function TriggerCard({ entry }: TriggerCardProps) {
  const { trigger, situational_context, compressed_summary } = entry;
  const [showSituational, setShowSituational] = useState(false);
  const [showCompressed, setShowCompressed] = useState(false);
  const getTriggerIcon = () => {
    switch (trigger.type) {
      case "user_input":
        return <User size={16} className={css({ color: "blue.500" })} />;
      case "wakeup":
        return <Coffee size={16} className={css({ color: "orange.500" })} />;
      default:
        return (
          <MessageCircle size={16} className={css({ color: "gray.500" })} />
        );
    }
  };

  const getTriggerReason = () => {
    switch (trigger.type) {
      case "user_input":
        return trigger.user_name;
      case "wakeup":
        return "Wakeup";
      default:
        const _exhaustiveCheck: never = trigger;
        return _exhaustiveCheck;
    }
  };

  const getTriggerContent = () => {
    switch (trigger.type) {
      case "user_input":
        return trigger.content;
      case "wakeup":
        return "Continuing to exist and experience...";
      default:
        return "";
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className={css({ fontSize: "xl", color: "gray.300" })}>
      {/* Header with icon/reason and timestamp */}
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        })}
      >
        <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
          {getTriggerIcon()}
          <span>{getTriggerReason()}</span>
        </div>
        <div
          className={css({
            display: "flex",
            alignItems: "center",
            gap: 1,
            fontSize: "base",
            color: "gray.400",
          })}
        >
          <Clock size={12} />
          <span>{formatTime(trigger.timestamp)}</span>
        </div>
      </div>

      {/* Content */}
      <div className={css({ color: "gray.200", whiteSpace: "pre-wrap" })}>
        {getTriggerContent()}
      </div>

      {/* Images */}
      {trigger.type === "user_input" &&
        trigger.image_urls &&
        trigger.image_urls.length > 0 && (
          <div
            className={css({
              mt: 3,
              display: "flex",
              flexWrap: "wrap",
              gap: 2,
            })}
          >
            {trigger.image_urls.map((imageUrl, index) => {
              return (
                <ImageDisplay
                  key={index}
                  src={imageUrl}
                  maxWidth="100px"
                  maxHeight="100px"
                  alt={`User provided image ${index + 1}`}
                  exactSize={true}
                />
              );
            })}
          </div>
        )}

      {/* Collapsible Debug Sections */}
      <div
        className={css({
          mt: 3,
          pt: 3,
          borderTop: "1px solid",
          borderTopColor: "gray.600",
        })}
      >
        {/* Situational Analysis */}
        {situational_context && (
          <>
            <button
              onClick={() => setShowSituational(!showSituational)}
              className={css({
                display: "flex",
                alignItems: "center",
                gap: 2,
                cursor: "pointer",
                background: "transparent",
                border: "none",
                color: "gray.400",
                fontSize: "sm",
                padding: 0,
                mb: 2,
                _hover: { color: "gray.300" },
              })}
            >
              {showSituational ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <span>Situational Analysis</span>
            </button>
            {showSituational && (
              <div
                className={css({
                  fontSize: "sm",
                  color: "gray.400",
                  whiteSpace: "pre-wrap",
                  pl: 4,
                  mb: 3,
                })}
              >
                {situational_context}
              </div>
            )}
          </>
        )}
        {/* Compressed Summary */}
        {compressed_summary && (
          <>
            <button
              onClick={() => setShowCompressed(!showCompressed)}
              className={css({
                display: "flex",
                alignItems: "center",
                gap: 2,
                cursor: "pointer",
                background: "transparent",
                border: "none",
                color: "gray.400",
                fontSize: "sm",
                padding: 0,
                mb: 2,
                _hover: { color: "gray.300" },
              })}
            >
              {showCompressed ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <span>Compressed Summary</span>
            </button>
            {showCompressed && (
              <div
                className={css({
                  fontSize: "sm",
                  color: "gray.400",
                  whiteSpace: "pre-wrap",
                  pl: 4,
                })}
              >
                {compressed_summary}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
