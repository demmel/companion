import { ActionResult } from "@/types";

export function isStreamingResult<T>(
  result: ActionResult<T>,
): result is { type: "streaming"; result: string } {
  return result.type === "streaming";
}

export function isFailureResult<T>(
  result: ActionResult<T>,
): result is { type: "failure"; error: string } {
  return result.type === "failure";
}

export function resultText<T>(
  result: ActionResult<T>,
  summarize: (content: T) => string,
): string {
  switch (result.type) {
    case "streaming":
      return result.result;
    case "failure":
      return result.error;
    case "success":
      return summarize(result.content);
  }
}
