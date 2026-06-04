import { css } from "@styled-system/css";
import { Loader2, Image as ImageIcon } from "lucide-react";
import { UpdateAppearanceAction, Action, ActionData } from "@/types";
import { ImageDisplay } from "../common/ImageDisplay";
import { useState } from "react";
import { isStreamingResult, resultText } from "./actionResult";

interface UpdateAppearanceActionDisplayProps {
  action: UpdateAppearanceAction;
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

type CompletedUpdateAppearanceAction = Extract<
  ActionData,
  { type: "update_appearance" }
>;
type SuccessfulUpdateAppearanceAction = CompletedUpdateAppearanceAction & {
  result: Extract<
    CompletedUpdateAppearanceAction["result"],
    { type: "success" }
  >;
};

interface RegenerateImageResponse {
  success: boolean;
  new_image_url?: string;
  error?: string;
}

function isCompletedUpdateAppearanceAction(
  action: UpdateAppearanceAction,
): action is SuccessfulUpdateAppearanceAction {
  return "input" in action && action.result.type === "success";
}

export function UpdateAppearanceActionDisplay({
  action,
  triggerId,
  actionIndex,
  updateAction,
}: UpdateAppearanceActionDisplayProps) {
  const [isRegenerating, setIsRegenerating] = useState(false);
  const isStreaming = isStreamingResult(action.result);
  const result = resultText(
    action.result,
    (content) =>
      `Appearance updated: ${content.new_appearance} (reason: ${content.reason})`,
  );
  const image =
    action.result.type === "success" ? action.result.content.image_result : null;
  const imageDescription =
    action.result.type === "success"
      ? action.result.content.image_description
      : "";

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

      const data: RegenerateImageResponse = await response.json();

      if (!data.success) {
        console.error("Failed to regenerate image:", data.error);
        // TODO: Show error toast/notification
      } else {
        // Update the action with the new image URL
        if (data.new_image_url && isCompletedUpdateAppearanceAction(action)) {
          const updatedAction: SuccessfulUpdateAppearanceAction = {
            ...action,
            result: {
              ...action.result,
              content: {
                ...action.result.content,
                image_result: {
                  ...action.result.content.image_result,
                  image_url: data.new_image_url,
                },
              },
            },
          };
          updateAction(triggerId, actionIndex, updatedAction);
        }
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
              color: "purple.500",
            })}
          />
        ) : (
          <ImageIcon size={16} className={css({ color: "purple.500" })} />
        )}
        <span
          className={css({
            fontSize: "xl",
            fontWeight: "medium",
            color: "gray.300",
          })}
        >
          {isStreaming ? "Updating appearance..." : "Appearance updated"}
        </span>
      </div>

      {/* Show image if available */}
      {image?.image_url ? (
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
            src={image.image_url}
            alt={imageDescription || "Agent appearance"}
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
            Generating new appearance...
          </div>
        ) : (
          result
        )}
      </div>
    </div>
  );
}
