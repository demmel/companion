import { css } from "@styled-system/css";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { TriggerHistoryEntry, Summary } from "@/types";
import { TriggerCard } from "./trigger/TriggerCard";
import { ActionDisplay } from "./action/ActionDisplay";
import { SummaryCard } from "./SummaryCard";

interface TimelineProps {
  triggerEntries: TriggerHistoryEntry[];
  summaries: Summary[];
  isStreamActive: boolean;
}

interface TimelineEntryProps {
  entry: TriggerHistoryEntry;
  isActive?: boolean;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
}

function TimelineEntry({
  entry,
  isActive = false,
  isExpanded = false,
  onToggleExpanded,
}: TimelineEntryProps) {
  const hasActions = entry.actions_taken.length > 0;
  const streamingCount = entry.actions_taken.filter(
    (action) => action.status.type === "streaming",
  ).length;
  const completedCount = entry.actions_taken.filter(
    (action) => action.status.type !== "streaming",
  ).length;

  return (
    <div
      className={css({
        mb: 4,
        border: "1px solid",
        borderColor: isActive ? "blue.600" : "gray.700",
        rounded: "lg",
        overflow: "hidden",
        bg: isActive ? "blue.900/20" : "gray.800",
      })}
    >
      {/* Trigger header */}
      <div
        className={css({
          p: 3,
          bg: isActive ? "blue.800/50" : "gray.700/30",
        })}
      >
        <TriggerCard trigger={entry.trigger} />

        {/* Actions summary and expand toggle */}
        {hasActions && (
          <div
            className={css({
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mt: 3,
              pt: 3,
              borderTop: "1px solid",
              borderColor: "gray.600",
            })}
          >
            <div
              className={css({
                fontSize: "sm",
                color: "gray.400",
              })}
            >
              {streamingCount > 0 && (
                <span
                  className={css({
                    color: "blue.400",
                  })}
                >
                  {streamingCount} streaming
                  {completedCount > 0 && ", "}
                </span>
              )}
              {completedCount > 0 && <span>{completedCount} completed</span>}
            </div>

            <button
              onClick={onToggleExpanded}
              className={css({
                display: "flex",
                alignItems: "center",
                gap: 1,
                fontSize: "sm",
                color: "blue.400",
                cursor: "pointer",
                _hover: { color: "blue.300" },
              })}
            >
              {isExpanded ? (
                <>
                  <ChevronDown size={16} />
                  Hide actions
                </>
              ) : (
                <>
                  <ChevronRight size={16} />
                  Show actions
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Actions list (expandable) - in original chronological order */}
      {hasActions && isExpanded && (
        <div
          className={css({
            p: 4,
            bg: "gray.900",
          })}
        >
          <div>
            {entry.actions_taken.map((action, index) => (
              <div
                key={index}
                className={css({
                  mb: index < entry.actions_taken.length - 1 ? 3 : 0,
                })}
              >
                <ActionDisplay action={action} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function Timeline({ triggerEntries, summaries, isStreamActive }: TimelineProps) {
  const [collapsedEntries, setCollapsedEntries] = useState<Set<string>>(
    new Set(),
  );

  const toggleExpanded = (entryId: string) => {
    setCollapsedEntries((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(entryId)) {
        newSet.delete(entryId);
      } else {
        newSet.add(entryId);
      }
      return newSet;
    });
  };

  // Merge summaries and entries in chronological order
  const timelineItems: Array<{ type: 'entry'; entry: TriggerHistoryEntry; index: number } | { type: 'summary'; summary: Summary }> = [];
  
  // Add entries with their original indices
  triggerEntries.forEach((entry, index) => {
    timelineItems.push({ type: 'entry', entry, index });
  });

  // Insert summaries at their correct positions
  summaries.forEach((summary) => {
    timelineItems.splice(summary.insert_at_index, 0, { type: 'summary', summary });
  });

  // The last entry is "active" if we're currently streaming
  const activeEntryIndex = isStreamActive ? triggerEntries.length - 1 : -1;

  return (
    <div
      className={css({
        maxW: "4xl",
        // mx: "auto",
      })}
    >
      {/* Timeline items */}
      <div>
        {timelineItems.map((item, index) => {
          if (item.type === 'summary') {
            return (
              <div
                key={`summary-${index}`}
                className={css({
                  mb: 4,
                  border: "1px solid",
                  borderColor: "yellow.700",
                  rounded: "lg",
                  overflow: "hidden",
                  bg: "yellow.900/10",
                })}
              >
                <div className={css({ p: 3, bg: "yellow.800/20" })}>
                  <SummaryCard summary={item.summary} />
                </div>
              </div>
            );
          } else {
            const isActive = item.index === activeEntryIndex;
            return (
              <TimelineEntry
                key={item.entry.entry_id}
                entry={item.entry}
                isActive={isActive}
                isExpanded={!collapsedEntries.has(item.entry.entry_id)}
                onToggleExpanded={() => toggleExpanded(item.entry.entry_id)}
              />
            );
          }
        })}
      </div>
    </div>
  );
}
