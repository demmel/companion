import { css } from "@styled-system/css";
import { FileText, Clock, Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { Summary } from "@/types";

interface SummaryCardProps {
  summary: Summary;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
}

export function SummaryCard({ summary, isExpanded = true, onToggleExpanded }: SummaryCardProps) {
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
    <div
      className={css({
        mb: 4,
        border: "1px solid",
        borderColor: "yellow.600",
        rounded: "lg",
        overflow: "hidden",
        bg: "yellow.900/10",
      })}
    >
      <div
        className={css({
          p: 3,
          bg: "yellow.800/20",
        })}
      >
        {/* Header with icon/reason and timestamp */}
        <div
          className={css({
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: isExpanded ? 3 : 0,
          })}
        >
          <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
            {isInProgress ? (
              <Loader2 size={16} className={css({ color: "yellow.500", animation: "spin 2s linear infinite" })} />
            ) : (
              <FileText size={16} className={css({ color: "yellow.500" })} />
            )}
            <span className={css({ fontSize: "lg", fontWeight: "medium", color: "gray.200" })}>
              {isInProgress ? "Summarizing..." : "Summary"}
            </span>
          </div>
          <div className={css({ display: "flex", alignItems: "center", gap: 3 })}>
            <div
              className={css({
                display: "flex",
                alignItems: "center",
                gap: 1,
                fontSize: "sm",
                color: "gray.400",
              })}
            >
              <Clock size={12} />
              <span>{formatTime(summary.created_at)}</span>
            </div>
            
            {onToggleExpanded && (
              <button
                onClick={onToggleExpanded}
                className={css({
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  fontSize: "sm",
                  color: "yellow.400",
                  cursor: "pointer",
                  _hover: { color: "yellow.300" },
                })}
              >
                {isExpanded ? (
                  <>
                    <ChevronDown size={16} />
                    <span>Hide</span>
                  </>
                ) : (
                  <>
                    <ChevronRight size={16} />
                    <span>Show</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Content - collapsible */}
        {isExpanded && (
          <div className={css({ 
            color: "gray.200",
            whiteSpace: "pre-wrap",
            lineHeight: 1.6,
          })}>
            {isInProgress 
              ? `Summarizing ${summary.messages_to_summarize} messages (keeping ${summary.recent_messages_kept} recent)...`
              : summary.summary_text
            }
          </div>
        )}
      </div>
    </div>
  );
}