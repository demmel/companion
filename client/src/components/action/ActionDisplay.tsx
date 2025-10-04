import { css } from "@styled-system/css";
import { Action } from "@/types";
import { ThinkActionDisplay } from "./ThinkActionDisplay";
import { SpeakActionDisplay } from "./SpeakActionDisplay";
import { UpdateAppearanceActionDisplay } from "./UpdateAppearanceActionDisplay";
import { UpdateEnvironmentActionDisplay } from "./UpdateEnvironmentActionDisplay";
import { UpdateMoodActionDisplay } from "./UpdateMoodActionDisplay";
import { WaitActionDisplay } from "./WaitActionDisplay";
import { CreativeInspirationActionDisplay } from "./CreativeInspirationActionDisplay";
import { AddPriorityActionDisplay } from "./AddPriorityActionDisplay";
import { RemovePriorityActionDisplay } from "./RemovePriorityActionDisplay";
import { FetchUrlActionDisplay } from "./FetchUrlActionDisplay";
import { SearchWebActionDisplay } from "./SearchWebActionDisplay";
import { useState } from "react";
import { ChevronDown } from "lucide-react";

interface ActionDisplayProps {
  action: Action;
}

export function ActionDisplay({ action }: ActionDisplayProps) {
  const [showDebugInfo, setShowDebugInfo] = useState(false);

  const renderActionContent = () => {
    switch (action.type) {
      case "think":
        return <ThinkActionDisplay action={action} />;
      case "speak":
        return <SpeakActionDisplay action={action} />;
      case "update_appearance":
        return <UpdateAppearanceActionDisplay action={action} />;
      case "update_environment":
        return <UpdateEnvironmentActionDisplay action={action} />;
      case "update_mood":
        return <UpdateMoodActionDisplay action={action} />;
      case "wait":
        return <WaitActionDisplay action={action} />;
      case "get_creative_inspiration":
        return <CreativeInspirationActionDisplay action={action} />;
      case "add_priority":
        return <AddPriorityActionDisplay action={action} />;
      case "remove_priority":
        return <RemovePriorityActionDisplay action={action} />;
      case "fetch_url":
        return <FetchUrlActionDisplay action={action} />;
      case "search_web":
        return <SearchWebActionDisplay action={action} />;
      default:
        return (
          <div className={css({ p: 2, color: "gray.500", fontSize: "sm" })}>
            Unknown action type: {(action as any).action_type}
          </div>
        );
    }
  };

  return (
    <div className={css({ mb: 2 })}>
      {renderActionContent()}
      <div className={css({ borderTop: "1px solid", borderColor: "gray.700" })}>
        <button
          onClick={() => setShowDebugInfo(!showDebugInfo)}
          className={css({
            w: "full",
            px: 3,
            py: 1,
            textAlign: "left",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "sm",
            color: "gray.500",
            _hover: { bg: "gray.800" },
          })}
        >
          <span>Why this action?</span>
          <ChevronDown
            size={12}
            className={css({
              transform: showDebugInfo ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
            })}
          />
        </button>

        {showDebugInfo && (
          <div
            className={css({
              px: 3,
              pb: 2,
              fontSize: "sm",
              color: "gray.400",
              fontStyle: "italic",
            })}
          >
            {action.context_given}
          </div>
        )}
      </div>
    </div>
  );
}
