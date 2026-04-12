import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { OllamaModelsPage } from "../OllamaModelsPage";
import type { AgentClient } from "@/client";

describe("OllamaModelsPage", () => {
  const client = {
    getInstalledOllamaModels: vi.fn(),
    getSupportedModels: vi.fn(),
    getModelConfig: vi.fn(),
    pullOllamaModel: vi.fn(),
    deleteOllamaModel: vi.fn(),
  } as unknown as AgentClient & {
    getInstalledOllamaModels: ReturnType<typeof vi.fn>;
    getSupportedModels: ReturnType<typeof vi.fn>;
    getModelConfig: ReturnType<typeof vi.fn>;
    pullOllamaModel: ReturnType<typeof vi.fn>;
    deleteOllamaModel: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    vi.clearAllMocks();
    client.getInstalledOllamaModels.mockResolvedValue({
      models: [
        {
          name: "mistral-small3.2:latest",
          size: 1024,
          modified_at: "2026-04-10T12:00:00",
          digest: null,
          details: {},
        },
      ],
    });
    client.getSupportedModels.mockResolvedValue({
      models: [
        "mistral-small3.2:latest",
        "llama3.1:8b",
        "claude-sonnet-4-5-20250929",
      ],
      ollama_models: ["mistral-small3.2:latest", "llama3.1:8b"],
    });
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
      tts_rewrite_model:
        "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    });
    client.pullOllamaModel.mockResolvedValue({
      name: "llama3.1:8b",
      message: "Pulled Ollama model 'llama3.1:8b'",
      timestamp: "2026-04-11T00:00:00",
    });
    client.deleteOllamaModel.mockResolvedValue({
      name: "mistral-small3.2:latest",
      message: "Deleted Ollama model 'mistral-small3.2:latest'",
      timestamp: "2026-04-11T00:00:00",
    });
  });

  it("loads installed models and allows pulling a supported Ollama model", async () => {
    const user = userEvent.setup();

    render(<OllamaModelsPage client={client} onBack={vi.fn()} />);

    expect(await screen.findByText("mistral-small3.2:latest")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /llama3.1:8b/i })).toBeInTheDocument();

    client.getInstalledOllamaModels.mockResolvedValueOnce({
      models: [
        {
          name: "mistral-small3.2:latest",
          size: 1024,
          modified_at: "2026-04-10T12:00:00",
          digest: null,
          details: {},
        },
        {
          name: "llama3.1:8b",
          size: 2048,
          modified_at: "2026-04-11T08:00:00",
          digest: null,
          details: {},
        },
      ],
    });

    await user.click(screen.getByRole("button", { name: /llama3.1:8b/i }));

    await waitFor(() => {
      expect(client.pullOllamaModel).toHaveBeenCalledWith("llama3.1:8b");
    });
    expect(
      await screen.findByText("Pulled Ollama model 'llama3.1:8b'"),
    ).toBeInTheDocument();
  });

  it("disables deletion for models that are still configured", async () => {
    client.getModelConfig.mockResolvedValueOnce({
      state_initialization_model: "mistral-small3.2:latest",
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

    render(<OllamaModelsPage client={client} onBack={vi.fn()} />);

    expect(await screen.findByText("In use: State Initialization")).toBeInTheDocument();
    expect(screen.getByText("In use: TTS Rewrite")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /delete/i })).toBeDisabled();
    expect(client.deleteOllamaModel).not.toHaveBeenCalled();
  });
});
