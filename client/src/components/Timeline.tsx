import { css } from "@styled-system/css";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { TriggerHistoryEntry, TimelineEntry } from "@/types";
import { TriggerCard } from "./trigger/TriggerCard";
import { ActionDisplay } from "./action/ActionDisplay";
import { SummaryCard } from "./SummaryCard";

interface TimelineProps {
  entries: TimelineEntry[];
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

interface TimelineItemProps {
  timelineEntry: TimelineEntry;
  isActive?: boolean;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
}

function TimelineItem({ timelineEntry, isActive, isExpanded, onToggleExpanded }: TimelineItemProps) {
  switch (timelineEntry.type) {
    case "trigger": {
      const entry = timelineEntry.entry;
      return (
        <TimelineEntry
          entry={entry}
          isActive={isActive}
          isExpanded={isExpanded}
          onToggleExpanded={onToggleExpanded}
        />
      );
    }
    case "summary": {
      return (
        <SummaryCard 
          summary={timelineEntry.summary} 
          isExpanded={isExpanded}
          onToggleExpanded={onToggleExpanded}
        />
      );
    }
    default: {
      // TypeScript ensures we handle all cases
      const _exhaustive: never = timelineEntry;
      return _exhaustive;
    }
  }
}

export function Timeline({ entries, isStreamActive }: TimelineProps) {
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

  // Find the last trigger entry for active streaming
  const triggerEntries = entries.filter((entry): entry is { type: "trigger"; entry: TriggerHistoryEntry } => entry.type === "trigger");
  const activeEntryIndex = isStreamActive ? triggerEntries.length - 1 : -1;

  return (
    <div
      className={css({
        maxW: "4xl",
        // mx: "auto",
      })}
    >
      <div>
        {entries.map((timelineEntry) => {
          // Determine if this trigger entry is active
          let isActive = false;
          let isExpanded = undefined;
          let onToggleExpanded = undefined;
          
          if (timelineEntry.type === "trigger") {
            const entry = timelineEntry.entry;
            const triggerIndex = triggerEntries.findIndex(te => te.entry.entry_id === entry.entry_id);
            isActive = triggerIndex === activeEntryIndex;
            isExpanded = !collapsedEntries.has(entry.entry_id);
            onToggleExpanded = () => toggleExpanded(entry.entry_id);
          } else if (timelineEntry.type === "summary") {
            const summaryKey = `summary-${timelineEntry.summary.insert_at_index}`;
            isExpanded = !collapsedEntries.has(summaryKey);
            onToggleExpanded = () => toggleExpanded(summaryKey);
          }

          return (
            <TimelineItem
              key={timelineEntry.type === "trigger" ? timelineEntry.entry.entry_id : `summary-${timelineEntry.summary.insert_at_index}`}
              timelineEntry={timelineEntry}
              isActive={isActive}
              isExpanded={isExpanded}
              onToggleExpanded={onToggleExpanded}
            />
          );
        })}
      </div>
    </div>
  );
}
