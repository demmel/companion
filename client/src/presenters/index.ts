import { ConversationPresenter } from "./types";
import { RoleplayPresenter } from "./RoleplayPresenter";
import { GenericPresenter } from "./GenericPresenter";

export function getPresenterForConfig(
  configName: string,
): ConversationPresenter {
  switch (configName) {
    case "roleplay":
      return RoleplayPresenter;
    case "coding":
    case "general":
    default:
      return GenericPresenter;
  }
}

export * from "./types";
