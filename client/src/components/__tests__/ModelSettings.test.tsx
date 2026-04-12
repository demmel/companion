import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModelSettings } from "../ModelSettings";
import type { AgentClient } from "@/client";

describe("ModelSettings", () => {
  const client = {
    getModelConfig: vi.fn(),
    getSupportedModels: vi.fn(),
    updateModelConfig: vi.fn(),
  } as unknown as AgentClient & {
    getModelConfig: ReturnType<typeof vi.fn>;
    getSupportedModels: ReturnType<typeof vi.fn>;
    updateModelConfig: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    vi.clearAllMocks();
    client.getModelConfig.mockResolvedValue({
      state_initialization_model: "claude-sonnet-4-5-20250929",
      action_planning_model: "claude-sonnet-4-5-20250929",
      situational_analysis_model: "claude-sonnet-4-5-20250929",
      memory_retrieval_model: "claude-sonnet-4-5-20250929",
      memory_formation_model: "claude-sonnet-4-5-20250929",
      trigger_compression_model: "claude-sonnet-4-5-20250929",
      think_action_model: "claude-sonnet-4-5-20250929",
      speak_action_model: "claude-sonnet-4-5-20250929",
      visual_action_model: "claude-sonnet-4-5-20250929",
      fetch_url_action_model: "claude-sonnet-4-5-20250929",
      evaluate_priorities_action_model: "claude-sonnet-4-5-20250929",
      tts_rewrite_model: "mistral-small3.2:latest",
    });
    client.getSupportedModels.mockResolvedValue({
      models: [
        "claude-sonnet-4-5-20250929",
        "mistral-small3.2:latest",
        "llama3.1:8b",
      ],
      ollama_models: ["mistral-small3.2:latest", "llama3.1:8b"],
    });
    client.updateModelConfig.mockResolvedValue({
      message: "Model configuration updated successfully",
      timestamp: "2026-04-12T00:00:00",
      config: {},
    });
  });

  it("saves an installed Ollama model selected from the dropdown", async () => {
    const user = userEvent.setup();

    render(<ModelSettings isOpen onClose={vi.fn()} client={client} />);

    // Wait for the form to load and find the first select showing Claude Sonnet
    const selects = await screen.findAllByDisplayValue("claude-sonnet-4-5-20250929");
    // Select an installed Ollama model in the first field (State Initialization)
    await user.selectOptions(selects[0], "mistral-small3.2:latest");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(client.updateModelConfig).toHaveBeenCalledTimes(1);
    });

    expect(client.updateModelConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        state_initialization_model: "mistral-small3.2:latest",
      }),
    );
  });

  it("shows all installed Ollama models and Anthropic models as options", async () => {
    render(<ModelSettings isOpen onClose={vi.fn()} client={client} />);

    await screen.findAllByDisplayValue("claude-sonnet-4-5-20250929");

    // All three models from getSupportedModels should appear as options
    expect(screen.getAllByRole("option", { name: "claude-sonnet-4-5-20250929" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("option", { name: "mistral-small3.2:latest" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("option", { name: "llama3.1:8b" }).length).toBeGreaterThan(0);
  });

  it("shows the current config value even when it is not in the suggestions list", async () => {
    client.getModelConfig.mockResolvedValue({
      state_initialization_model: "custom-org/custom-reasoner:latest",
      action_planning_model: "claude-sonnet-4-5-20250929",
      situational_analysis_model: "claude-sonnet-4-5-20250929",
      memory_retrieval_model: "claude-sonnet-4-5-20250929",
      memory_formation_model: "claude-sonnet-4-5-20250929",
      trigger_compression_model: "claude-sonnet-4-5-20250929",
      think_action_model: "claude-sonnet-4-5-20250929",
      speak_action_model: "claude-sonnet-4-5-20250929",
      visual_action_model: "claude-sonnet-4-5-20250929",
      fetch_url_action_model: "claude-sonnet-4-5-20250929",
      evaluate_priorities_action_model: "claude-sonnet-4-5-20250929",
      tts_rewrite_model: "mistral-small3.2:latest",
    });

    render(<ModelSettings isOpen onClose={vi.fn()} client={client} />);

    // The custom model not in suggestions list should still be shown as selected
    const select = await screen.findByDisplayValue("custom-org/custom-reasoner:latest");
    expect(select).toBeDefined();
  });
});
