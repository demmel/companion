import { css } from "@styled-system/css";
import { Loader2, MapPin } from "lucide-react";
import { UpdateEnvironmentAction, Action } from "@/types";
import { ImageDisplay } from "../common/ImageDisplay";
import { useState } from "react";

interface UpdateEnvironmentActionDisplayProps {
  action: UpdateEnvironmentAction;
  // TODO: Long-term, consider passing an onRegenerate callback instead of triggerId
  // to improve encapsulation (action display shouldn't need to know about triggers)
  triggerId: string;
  actionIndex: number;
  updateAction: (
    triggerId: string,
    actionIndex: number,
    updates: Partial<Action>,
  ) => void;
}

export function UpdateEnvironmentActionDisplay({
  action,
  triggerId,
  actionIndex,
  updateAction,
}: UpdateEnvironmentActionDisplayProps) {
  const [isRegenerating, setIsRegenerating] = useState(false);
  const isStreaming = action.status.type === "streaming";
  const result =
    action.status.type === "error"
      ? `Error: ${action.status.error}`
      : action.status.result;

  const handleRegenerate = async () => {
    setIsRegenerating(true);
    try {
      const response = await fetch("/api/regenerate-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          trigger_id: triggerId,
          action_index: actionIndex,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        console.error("Failed to regenerate image:", data.error);
        // TODO: Show error toast/notification
      } else {
        // Update the action with the new image URL
        updateAction(triggerId, actionIndex, {
          image_url: data.new_image_url,
        });
      }
    } catch (error) {
      console.error("Error regenerating image:", error);
      // TODO: Show error toast/notification
    } finally {
      setIsRegenerating(false);
    }
  };

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
              color: "green.500",
            })}
          />
        ) : (
          <MapPin size={16} className={css({ color: "green.500" })} />
        )}
        <span
          className={css({
            fontSize: "xl",
            fontWeight: "medium",
            color: "gray.300",
          })}
        >
          {isStreaming ? "Updating environment..." : "Environment updated"}
        </span>
      </div>

      {/* Show image if available */}
      {action.image_url ? (
        <div className={css({ mb: 3, position: "relative" })}>
          {isRegenerating && (
            <div
              className={css({
                position: "absolute",
                inset: 0,
                bg: "rgba(0, 0, 0, 0.5)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: 10,
                rounded: "md",
              })}
            >
              <Loader2
                size={32}
                className={css({
                  animation: "spin 1s linear infinite",
                  color: "white",
                })}
              />
            </div>
          )}
          <ImageDisplay
            src={action.image_url}
            alt={action.image_description || "Agent environment"}
            maxWidth="100%"
            maxHeight="300px"
            onRegenerate={handleRegenerate}
          />
        </div>
      ) : (
        isStreaming && (
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
              <div className={css({ fontSize: "xl" })}>Generating image...</div>
            </div>
          </div>
        )
      )}

      {/* Show description */}
      <div
        className={css({
          fontSize: "xl",
          color: "gray.400",
        })}
      >
        {isStreaming && !result ? (
          <div className={css({ fontStyle: "italic" })}>
            {action.context_given || "Generating new environment..."}
          </div>
        ) : (
          result
        )}
      </div>
    </div>
  );
}
