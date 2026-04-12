import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatHeader } from "../ChatHeader";
import type { AgentClient } from "@/client";

vi.mock("../AutoWakeupToggle", () => ({
  AutoWakeupToggle: () => <div>AutoWakeupToggle</div>,
}));

vi.mock("../UsernameSettings", () => ({
  UsernameSettings: () => null,
}));

vi.mock("../ModelSettings", () => ({
  ModelSettings: () => null,
}));

describe("ChatHeader", () => {
  it("opens the manage Ollama models page from the menu", async () => {
    const user = userEvent.setup();
    const onNavigateToOllamaModels = vi.fn();

    render(
      <ChatHeader
        client={{} as AgentClient}
        onNavigateToOllamaModels={onNavigateToOllamaModels}
      />,
    );

    await user.click(screen.getAllByRole("button")[0]);
    await user.click(screen.getByRole("button", { name: /manage ollama models/i }));

    expect(onNavigateToOllamaModels).toHaveBeenCalledTimes(1);
  });
});
