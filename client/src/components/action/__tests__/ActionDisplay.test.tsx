import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ActionDisplay } from "../ActionDisplay";
import { SpeakActionDisplay } from "../SpeakActionDisplay";
import { UpdateAppearanceActionDisplay } from "../UpdateAppearanceActionDisplay";
import { UpdateMoodActionDisplay } from "../UpdateMoodActionDisplay";
import type {
  Action,
  SpeakAction,
  UpdateAppearanceAction,
  UpdateMoodAction,
} from "@/types";

vi.mock("@/hooks/useActionAudio", () => ({
  useActionAudio: () => ({
    playState: "idle",
    handlePlayClick: vi.fn(),
  }),
}));

const streamingMoodAction = {
  type: "update_mood",
  reasoning: "Internal plan should stay in the why panel.",
  result: { type: "streaming", result: "" },
  duration_ms: 0,
  start_timestamp: "2024-01-01T10:00:00Z",
} satisfies UpdateMoodAction;

describe("action displays", () => {
  it("keeps reasoning out of action body while streaming and shows it only in why", async () => {
    const user = userEvent.setup();

    render(
      <ActionDisplay
        action={streamingMoodAction}
        triggerId="entry_1"
        actionIndex={0}
        updateAction={vi.fn()}
      />,
    );

    expect(screen.getByText("Adjusting mood...")).toBeInTheDocument();
    expect(
      screen.queryByText("Internal plan should stay in the why panel."),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /why this action/i }));

    expect(
      screen.getByText("Internal plan should stay in the why panel."),
    ).toBeInTheDocument();
  });

  it("renders speech success, failure, and streaming results from the new result envelope", () => {
    const successAction = {
      type: "speak",
      reasoning: "Speak plainly.",
      input: { intent: "Greet the user", tone: null },
      result: {
        type: "success",
        content: { response: "Hello from nested content." },
      },
      duration_ms: 25,
      start_timestamp: "2024-01-01T10:00:00Z",
    } satisfies SpeakAction;

    const failureAction = {
      ...successAction,
      result: { type: "failure", error: "Backend supplied failure." },
    } satisfies SpeakAction;

    const streamingAction = {
      type: "speak",
      reasoning: "Streaming should use partial text.",
      result: { type: "streaming", result: "Partial response" },
      duration_ms: 0,
      start_timestamp: "2024-01-01T10:00:00Z",
    } satisfies SpeakAction;

    const { rerender } = render(
      <SpeakActionDisplay
        action={successAction}
        triggerId="entry_1"
        actionIndex={0}
      />,
    );
    expect(screen.getByText("Hello from nested content.")).toBeInTheDocument();
    expect(screen.getByText("Listen")).toBeInTheDocument();

    rerender(
      <SpeakActionDisplay
        action={failureAction}
        triggerId="entry_1"
        actionIndex={0}
      />,
    );
    expect(screen.getByText("Backend supplied failure.")).toBeInTheDocument();
    expect(screen.queryByText("Listen")).not.toBeInTheDocument();

    rerender(
      <SpeakActionDisplay
        action={streamingAction}
        triggerId="entry_1"
        actionIndex={0}
      />,
    );
    expect(screen.getByText(/Partial response/)).toBeInTheDocument();
    expect(screen.queryByText("Listen")).not.toBeInTheDocument();
  });

  it("renders update mood success and streaming states from the generated action shape", () => {
    const successAction = {
      type: "update_mood",
      reasoning: "Mood should move.",
      input: {
        reason: "The user asked for a brighter tone.",
        new_mood: "happy",
        intensity: "medium",
      },
      result: {
        type: "success",
        content: {
          old_mood: "neutral",
          old_intensity: "low",
          new_mood: "happy",
          new_intensity: "medium",
          reason: "The user asked for a brighter tone.",
        },
      },
      duration_ms: 50,
      start_timestamp: "2024-01-01T10:00:00Z",
    } satisfies UpdateMoodAction;

    const { rerender } = render(
      <UpdateMoodActionDisplay action={successAction} />,
    );
    expect(
      screen.getByText(
        "Mood changed from neutral (low) to happy (medium): The user asked for a brighter tone.",
      ),
    ).toBeInTheDocument();

    rerender(<UpdateMoodActionDisplay action={streamingMoodAction} />);
    expect(screen.getByText("Adjusting mood...")).toBeInTheDocument();
    expect(
      screen.queryByText("Internal plan should stay in the why panel."),
    ).not.toBeInTheDocument();
  });

  it("renders update appearance from nested image result and does not use reasoning as pending copy", () => {
    const successAction = {
      type: "update_appearance",
      reasoning: "Internal image plan.",
      input: {
        reason: "Refresh the portrait.",
        change_description: "Silver jacket and warm lighting.",
      },
      result: {
        type: "success",
        content: {
          image_description: "Silver jacket and warm lighting.",
          old_appearance: "Old look",
          new_appearance: "New silver jacket look",
          reason: "Refresh the portrait.",
          image_result: {
            type: "image_generated",
            prompt: "Silver jacket and warm lighting.",
            chunks: null,
            image_path: "portrait.png",
            image_url: "/generated_images/portrait.png",
            width: 512,
            height: 512,
            num_inference_steps: 20,
            guidance_scale: 7.5,
            negative_prompt: null,
            original_description: null,
            optimization_confidence: null,
            camera_angle: null,
            viewpoint: null,
            optimization_notes: null,
          },
        },
      },
      duration_ms: 2000,
      start_timestamp: "2024-01-01T10:00:00Z",
    } satisfies UpdateAppearanceAction;

    const streamingAction = {
      type: "update_appearance",
      reasoning: "Internal image plan.",
      result: { type: "streaming", result: "" },
      duration_ms: 0,
      start_timestamp: "2024-01-01T10:00:00Z",
    } satisfies UpdateAppearanceAction;

    const updateAction =
      vi.fn<
        (triggerId: string, index: number, updates: Partial<Action>) => void
      >();

    const { rerender } = render(
      <UpdateAppearanceActionDisplay
        action={successAction}
        triggerId="entry_1"
        actionIndex={0}
        updateAction={updateAction}
      />,
    );

    expect(
      screen.getByText(
        "Appearance updated: New silver jacket look (reason: Refresh the portrait.)",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Silver jacket and warm lighting." }),
    ).toHaveAttribute("src", "/generated_images/portrait.png");

    rerender(
      <UpdateAppearanceActionDisplay
        action={streamingAction}
        triggerId="entry_1"
        actionIndex={0}
        updateAction={updateAction}
      />,
    );
    expect(screen.getByText("Generating new appearance...")).toBeInTheDocument();
    expect(screen.queryByText("Internal image plan.")).not.toBeInTheDocument();
  });
});
