import { css } from "@styled-system/css";
import { Action } from "@/types";
import { ThinkActionDisplay } from "./ThinkActionDisplay";
import { SpeakActionDisplay } from "./SpeakActionDisplay";
import { UpdateAppearanceActionDisplay } from "./UpdateAppearanceActionDisplay";
import { UpdateMoodActionDisplay } from "./UpdateMoodActionDisplay";
import { WaitActionDisplay } from "./WaitActionDisplay";
import { AddPriorityActionDisplay } from "./AddPriorityActionDisplay";
import { RemovePriorityActionDisplay } from "./RemovePriorityActionDisplay";

interface ActionDisplayProps {
  action: Action;
}

export function ActionDisplay({ action }: ActionDisplayProps) {
  const renderActionContent = () => {
    switch (action.type) {
      case "think":
        return <ThinkActionDisplay action={action} />;
      case "speak":
        return <SpeakActionDisplay action={action} />;
      case "update_appearance":
        return <UpdateAppearanceActionDisplay action={action} />;
      case "update_mood":
        return <UpdateMoodActionDisplay action={action} />;
      case "wait":
        return <WaitActionDisplay action={action} />;
      case "add_priority":
        return <AddPriorityActionDisplay action={action} />;
      case "remove_priority":
        return <RemovePriorityActionDisplay action={action} />;
      default:
        return (
          <div className={css({ p: 2, color: "gray.500", fontSize: "sm" })}>
            Unknown action type: {(action as any).action_type}
          </div>
        );
    }
  };

  return <div className={css({ mb: 2 })}>{renderActionContent()}</div>;
}
