import { css } from "@styled-system/css";
import { Loader2, Image as ImageIcon } from "lucide-react";
import { UpdateAppearanceAction } from "@/types";

interface UpdateAppearanceActionDisplayProps {
  action: UpdateAppearanceAction;
}

export function UpdateAppearanceActionDisplay({ action }: UpdateAppearanceActionDisplayProps) {
  const isStreaming = action.status.type === "streaming";
  const result = action.status.type === "error" ? 
    `Error: ${action.status.error}` : 
    action.status.result;
  return (
    <div
      className={css({
        border: "1px solid",
        borderColor: "gray.700",
        rounded: "lg",
        p: 3,
        mb: 2,
        bg: "gray.800",
      })}
    >
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          gap: 2,
          mb: 3,
        })}
      >
        {isStreaming ? (
          <Loader2
            size={16}
            className={css({
              animation: "spin 1s linear infinite",
              color: "purple.500",
            })}
          />
        ) : (
          <ImageIcon size={16} className={css({ color: "purple.500" })} />
        )}
        <span
          className={css({
            fontSize: "sm",
            fontWeight: "medium",
            color: "gray.300",
          })}
        >
          {isStreaming ? "Updating appearance..." : "Appearance updated"}
        </span>
      </div>

      {/* Show image if available */}
      {action.image_url && (
        <div className={css({ mb: 3 })}>
          <img
            src={action.image_url}
            alt={action.image_description || "Agent appearance"}
            className={css({
              maxW: "full",
              h: "auto",
              rounded: "lg",
              border: "1px solid",
              borderColor: "gray.600",
            })}
            style={{ maxHeight: "300px" }}
          />
        </div>
      )}

      {/* Show description */}
      <div
        className={css({
          fontSize: "sm",
          color: "gray.400",
        })}
      >
        {isStreaming && !result ? (
          <div className={css({ fontStyle: "italic" })}>
            {action.context_given || "Generating new appearance..."}
          </div>
        ) : (
          result
        )}
      </div>

      {/* Loading state for image generation */}
      {isStreaming && !action.image_url && (
        <div
          className={css({
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            h: "200px",
            bg: "gray.900",
            rounded: "lg",
            border: "2px dashed",
            borderColor: "gray.600",
          })}
        >
          <div className={css({ textAlign: "center", color: "gray.400" })}>
            <Loader2
              size={24}
              className={css({
                animation: "spin 1s linear infinite",
                mx: "auto",
                mb: 2,
              })}
            />
            <div className={css({ fontSize: "sm" })}>Generating image...</div>
          </div>
        </div>
      )}
    </div>
  );
}
