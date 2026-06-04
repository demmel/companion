import { css } from "@styled-system/css";
import { CreativeInspirationAction } from "@/types";

interface CreativeInspirationActionDisplayProps {
  action: CreativeInspirationAction;
}

export function CreativeInspirationActionDisplay({
  action,
}: CreativeInspirationActionDisplayProps) {
  const isError = action.result.type === "failure";
  const words =
    action.result.type === "success" ? action.result.content.words : [];

  return (
    <div
      className={css({
        p: 3,
        bg: "gray.900",
        borderLeft: "3px solid",
        borderColor: isError ? "red.500" : "purple.500",
      })}
    >
      <div
        className={css({
          fontSize: "sm",
          fontWeight: "medium",
          color: "purple.400",
          mb: 2,
        })}
      >
        💡 Creative Inspiration
      </div>

      {action.result.type === "success" && words.length > 0 && (
        <div className={css({ display: "flex", flexWrap: "wrap", gap: 2 })}>
          {words.map((word, index) => (
            <span
              key={index}
              className={css({
                px: 2,
                py: 1,
                bg: "purple.900",
                color: "purple.200",
                borderRadius: "md",
                fontSize: "sm",
                border: "1px solid",
                borderColor: "purple.700",
              })}
            >
              {word}
            </span>
          ))}
        </div>
      )}

      {isError && (
        <div className={css({ color: "red.400", fontSize: "sm" })}>
          {action.result.type === "failure"
            ? action.result.error
            : ""}
        </div>
      )}
    </div>
  );
}
