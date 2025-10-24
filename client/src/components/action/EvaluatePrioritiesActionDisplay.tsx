import { css } from "@styled-system/css";
import {
  Loader2,
  ListChecks,
  Plus,
  Trash2,
  GitMerge,
  RefreshCw,
  ArrowUpDown,
} from "lucide-react";
import { EvaluatePrioritiesAction, PriorityOperationResult } from "@/types";

interface EvaluatePrioritiesActionDisplayProps {
  action: EvaluatePrioritiesAction;
}

function OperationIcon({ type }: { type: PriorityOperationResult["operation_type"] }) {
  const iconProps = { size: 14, className: css({ flexShrink: 0 }) };

  switch (type) {
    case "add":
      return <Plus {...iconProps} className={css({ color: "green.400" })} />;
    case "remove":
      return <Trash2 {...iconProps} className={css({ color: "red.400" })} />;
    case "merge":
      return <GitMerge {...iconProps} className={css({ color: "blue.400" })} />;
    case "refine":
      return <RefreshCw {...iconProps} className={css({ color: "yellow.400" })} />;
    case "reorder":
      return <ArrowUpDown {...iconProps} className={css({ color: "purple.400" })} />;
  }
}

export function EvaluatePrioritiesActionDisplay({
  action,
}: EvaluatePrioritiesActionDisplayProps) {
  const isStreaming = action.status.type === "streaming";
  const hasOperations = action.operations && action.operations.length > 0;

  return (
    <div
      className={css({
        display: "flex",
        flexDirection: "column",
        gap: 2,
        p: 3,
        bg: "purple.900/20",
        border: "1px solid",
        borderColor: "purple.700",
        rounded: "md",
        fontSize: "sm",
      })}
    >
      {/* Header */}
      <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
        {isStreaming ? (
          <Loader2
            size={16}
            className={css({
              animation: "spin 1s linear infinite",
              color: "purple.500",
            })}
          />
        ) : (
          <ListChecks size={16} className={css({ color: "purple.500" })} />
        )}

        <div className={css({ flex: 1 })}>
          <div className={css({ color: "purple.300", fontWeight: "medium" })}>
            {isStreaming
              ? "Evaluating priorities..."
              : action.status.type === "error"
                ? "Evaluation failed"
                : hasOperations
                  ? `Applied ${action.operations.length} operation${action.operations.length === 1 ? "" : "s"}`
                  : "No changes needed"}
          </div>
        </div>
      </div>

      {/* Operations list */}
      {hasOperations && !isStreaming && action.status.type !== "error" && (
        <div
          className={css({
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
            pl: 2,
          })}
        >
          {action.operations.map((op, index) => (
            <div
              key={index}
              className={css({
                display: "flex",
                alignItems: "flex-start",
                gap: 2,
                p: 2,
                bg: "purple.950/50",
                border: "1px solid",
                borderColor: "purple.800",
                rounded: "sm",
              })}
            >
              <OperationIcon type={op.operation_type} />
              <div
                className={css({
                  flex: 1,
                  color: "purple.100",
                  fontSize: "xs",
                  lineHeight: "relaxed",
                })}
              >
                {op.summary}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Error message */}
      {action.status.type === "error" && (
        <div
          className={css({
            color: "red.300",
            fontSize: "sm",
            p: 2,
            bg: "red.950/50",
            border: "1px solid",
            borderColor: "red.800",
            rounded: "sm",
          })}
        >
          {action.status.error}
        </div>
      )}
    </div>
  );
}
