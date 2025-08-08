import { css } from "@styled-system/css";
import { FileText, Clock, Loader2 } from "lucide-react";
import { Summary } from "@/types";

interface SummaryCardProps {
  summary: Summary;
}

export function SummaryCard({ summary }: SummaryCardProps) {
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

  const isInProgress = summary.status === "in_progress";

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
          {isInProgress ? (
            <Loader2 size={16} className={css({ color: "yellow.500", animation: "spin 2s linear infinite" })} />
          ) : (
            <FileText size={16} className={css({ color: "yellow.500" })} />
          )}
          <span>{isInProgress ? "Summarizing..." : "Summary"}</span>
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
          <span>{formatTime(summary.created_at)}</span>
        </div>
      </div>

      {/* Content */}
      <div className={css({ color: "gray.200" })}>
        {isInProgress 
          ? `Summarizing ${summary.messages_to_summarize} messages (keeping ${summary.recent_messages_kept} recent)...`
          : summary.summary_text
        }
      </div>
    </div>
  );
}