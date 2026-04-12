import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Download, RefreshCcw, Trash2 } from "lucide-react";
import { css } from "@styled-system/css";
import { AgentClient } from "@/client";
import {
  InstalledOllamaModel,
  ModelConfigResponse,
  SupportedModelsResponse,
} from "@/types";

interface OllamaModelsPageProps {
  client: AgentClient;
  onBack: () => void;
}

const MODEL_LABELS: Record<keyof ModelConfigResponse, string> = {
  state_initialization_model: "State Initialization",
  action_planning_model: "Action Planning",
  situational_analysis_model: "Situational Analysis",
  memory_retrieval_model: "Memory Retrieval",
  memory_formation_model: "Memory Formation",
  trigger_compression_model: "Trigger Compression",
  think_action_model: "Think Action",
  speak_action_model: "Speak Action",
  visual_action_model: "Visual Actions",
  fetch_url_action_model: "Fetch URL Action",
  evaluate_priorities_action_model: "Evaluate Priorities",
  tts_rewrite_model: "TTS Rewrite",
};

function formatBytes(size: number | null): string {
  if (size === null || Number.isNaN(size)) {
    return "Unknown size";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = size;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatModifiedAt(modifiedAt: string | null): string {
  if (!modifiedAt) {
    return "Unknown update time";
  }

  const date = new Date(modifiedAt);
  if (Number.isNaN(date.getTime())) {
    return modifiedAt;
  }

  return date.toLocaleString();
}

function buildUsageMap(config: ModelConfigResponse | null): Map<string, string[]> {
  const usageMap = new Map<string, string[]>();
  if (!config) {
    return usageMap;
  }

  (Object.entries(MODEL_LABELS) as Array<[keyof ModelConfigResponse, string]>).forEach(
    ([field, label]) => {
      const modelName = config[field];
      const existing = usageMap.get(modelName) ?? [];
      existing.push(label);
      usageMap.set(modelName, existing);
    },
  );

  return usageMap;
}

export function OllamaModelsPage({ client, onBack }: OllamaModelsPageProps) {
  const [installedModels, setInstalledModels] = useState<InstalledOllamaModel[]>([]);
  const [supportedModels, setSupportedModels] = useState<SupportedModelsResponse | null>(
    null,
  );
  const [config, setConfig] = useState<ModelConfigResponse | null>(null);
  const [customModelName, setCustomModelName] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [busyModelName, setBusyModelName] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadData = useCallback(
    async (isManualRefresh: boolean = false) => {
      if (isManualRefresh) {
        setIsRefreshing(true);
      } else {
        setIsLoading(true);
      }

      setErrorMessage(null);

      try {
        const [installedResponse, supportedResponse, configResponse] =
          await Promise.all([
            client.getInstalledOllamaModels(),
            client.getSupportedModels(),
            client.getModelConfig(),
          ]);
        setInstalledModels(installedResponse.models);
        setSupportedModels(supportedResponse);
        setConfig(configResponse);
      } catch (error) {
        setErrorMessage(`Failed to load Ollama models: ${error}`);
      } finally {
        setIsLoading(false);
        setIsRefreshing(false);
      }
    },
    [client],
  );

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const usageMap = buildUsageMap(config);
  const installedModelNames = new Set(installedModels.map((model) => model.name));
  const availableSupportedModels =
    supportedModels?.ollama_models.filter((model) => !installedModelNames.has(model)) ?? [];

  const handlePull = async (modelName: string) => {
    setBusyModelName(modelName);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const response = await client.pullOllamaModel(modelName);
      setSuccessMessage(response.message);
      setCustomModelName("");
      await loadData(true);
    } catch (error) {
      setErrorMessage(`Failed to pull model '${modelName}': ${error}`);
    } finally {
      setBusyModelName(null);
    }
  };

  const handleDelete = async (modelName: string) => {
    setBusyModelName(modelName);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const response = await client.deleteOllamaModel(modelName);
      setSuccessMessage(response.message);
      await loadData(true);
    } catch (error) {
      setErrorMessage(`Failed to delete model '${modelName}': ${error}`);
    } finally {
      setBusyModelName(null);
    }
  };

  const handleCustomPull = async () => {
    const trimmedName = customModelName.trim();
    if (!trimmedName) {
      setErrorMessage("Enter an Ollama model name to pull.");
      return;
    }

    await handlePull(trimmedName);
  };

  return (
    <div
      className={css({
        minHeight: "100vh",
        bg: "gray.900",
        color: "gray.100",
      })}
    >
      <div
        className={css({
          maxW: "6xl",
          mx: "auto",
          px: 4,
          py: 6,
        })}
      >
        <div
          className={css({
            display: "flex",
            flexDirection: { base: "column", md: "row" },
            alignItems: { base: "stretch", md: "center" },
            justifyContent: "space-between",
            gap: 4,
            mb: 6,
          })}
        >
          <div>
            <button
              onClick={onBack}
              className={css({
                display: "inline-flex",
                alignItems: "center",
                gap: 2,
                mb: 3,
                px: 3,
                py: 2,
                rounded: "md",
                color: "gray.300",
                bg: "gray.800",
                _hover: { bg: "gray.700", color: "white" },
              })}
            >
              <ArrowLeft size={16} />
              Back to Chat
            </button>
            <h1
              className={css({
                fontSize: "3xl",
                fontWeight: "semibold",
              })}
            >
              Manage Ollama Models
            </h1>
            <p className={css({ mt: 2, color: "gray.400", maxW: "2xl" })}>
              Review installed models, pull new ones, and remove models that are
              no longer in use.
            </p>
          </div>

          <button
            onClick={() => void loadData(true)}
            disabled={isRefreshing || isLoading || busyModelName !== null}
            className={css({
              alignSelf: { base: "stretch", md: "flex-start" },
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 2,
              px: 4,
              py: 2.5,
              rounded: "md",
              bg: "gray.800",
              color: "gray.200",
              border: "1px solid",
              borderColor: "gray.700",
              _hover: { bg: "gray.700", color: "white" },
              _disabled: { opacity: 0.5, cursor: "not-allowed" },
            })}
          >
            <RefreshCcw size={16} />
            {isRefreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {errorMessage && (
          <div
            className={css({
              mb: 4,
              px: 4,
              py: 3,
              rounded: "lg",
              bg: "red.950",
              border: "1px solid",
              borderColor: "red.800",
              color: "red.200",
            })}
          >
            {errorMessage}
          </div>
        )}

        {successMessage && (
          <div
            className={css({
              mb: 4,
              px: 4,
              py: 3,
              rounded: "lg",
              bg: "green.950",
              border: "1px solid",
              borderColor: "green.800",
              color: "green.200",
            })}
          >
            {successMessage}
          </div>
        )}

        {isLoading ? (
          <div
            className={css({
              px: 4,
              py: 8,
              textAlign: "center",
              rounded: "xl",
              bg: "gray.800",
              border: "1px solid",
              borderColor: "gray.700",
              color: "gray.400",
            })}
          >
            Loading installed models...
          </div>
        ) : (
          <div
            className={css({
              display: "grid",
              gridTemplateColumns: { base: "1fr", xl: "2fr 1fr" },
              gap: 6,
            })}
          >
            <section
              className={css({
                rounded: "xl",
                bg: "gray.800",
                border: "1px solid",
                borderColor: "gray.700",
                p: 5,
              })}
            >
              <div
                className={css({
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 4,
                  mb: 4,
                })}
              >
                <div>
                  <h2 className={css({ fontSize: "xl", fontWeight: "medium" })}>
                    Installed Models
                  </h2>
                  <p className={css({ mt: 1, color: "gray.400", fontSize: "sm" })}>
                    {installedModels.length} model
                    {installedModels.length === 1 ? "" : "s"} currently available
                    in Ollama.
                  </p>
                </div>
              </div>

              {installedModels.length === 0 ? (
                <div
                  className={css({
                    px: 4,
                    py: 6,
                    rounded: "lg",
                    bg: "gray.900",
                    border: "1px dashed",
                    borderColor: "gray.700",
                    color: "gray.400",
                  })}
                >
                  No Ollama models are installed yet.
                </div>
              ) : (
                <div className={css({ display: "grid", gap: 3 })}>
                  {installedModels.map((model) => {
                    const usage = usageMap.get(model.name) ?? [];
                    const isBusy = busyModelName === model.name;
                    const deleteDisabled = usage.length > 0 || isBusy;

                    return (
                      <article
                        key={model.name}
                        className={css({
                          rounded: "lg",
                          bg: "gray.900",
                          border: "1px solid",
                          borderColor: "gray.700",
                          p: 4,
                        })}
                      >
                        <div
                          className={css({
                            display: "flex",
                            flexDirection: { base: "column", md: "row" },
                            alignItems: { base: "stretch", md: "flex-start" },
                            justifyContent: "space-between",
                            gap: 3,
                          })}
                        >
                          <div>
                            <div
                              className={css({
                                fontSize: "md",
                                fontWeight: "medium",
                                wordBreak: "break-word",
                              })}
                            >
                              {model.name}
                            </div>
                            <div
                              className={css({
                                mt: 2,
                                display: "flex",
                                flexWrap: "wrap",
                                gap: 2,
                                color: "gray.400",
                                fontSize: "sm",
                              })}
                            >
                              <span>{formatBytes(model.size)}</span>
                              <span>{formatModifiedAt(model.modified_at)}</span>
                            </div>
                            {usage.length > 0 && (
                              <div
                                className={css({
                                  mt: 3,
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 2,
                                })}
                              >
                                {usage.map((label) => (
                                  <span
                                    key={label}
                                    className={css({
                                      px: 2,
                                      py: 1,
                                      rounded: "full",
                                      bg: "blue.950",
                                      color: "blue.200",
                                      fontSize: "xs",
                                    })}
                                  >
                                    In use: {label}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>

                          <button
                            onClick={() => void handleDelete(model.name)}
                            disabled={deleteDisabled}
                            className={css({
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: 2,
                              minW: { base: "full", md: "9rem" },
                              px: 3,
                              py: 2,
                              rounded: "md",
                              bg: "red.950",
                              color: "red.200",
                              border: "1px solid",
                              borderColor: "red.800",
                              _hover: {
                                bg: deleteDisabled ? "red.950" : "red.900",
                              },
                              _disabled: {
                                opacity: 0.45,
                                cursor: "not-allowed",
                              },
                            })}
                          >
                            <Trash2 size={16} />
                            {isBusy ? "Deleting..." : "Delete"}
                          </button>
                        </div>

                        {usage.length > 0 && (
                          <p
                            className={css({
                              mt: 3,
                              color: "amber.200",
                              fontSize: "sm",
                            })}
                          >
                            Update model configuration before deleting this model.
                          </p>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            <section
              className={css({
                rounded: "xl",
                bg: "gray.800",
                border: "1px solid",
                borderColor: "gray.700",
                p: 5,
                alignSelf: "start",
              })}
            >
              <h2 className={css({ fontSize: "xl", fontWeight: "medium", mb: 2 })}>
                Pull Models
              </h2>
              <p className={css({ color: "gray.400", fontSize: "sm", mb: 4 })}>
                Install a supported Ollama model or paste any model identifier you
                want Ollama to fetch. Pulls complete before the page refreshes.
              </p>

              <div className={css({ display: "grid", gap: 3, mb: 5 })}>
                <label className={css({ fontSize: "sm", color: "gray.300" })}>
                  Custom model name
                </label>
                <input
                  value={customModelName}
                  onChange={(event) => setCustomModelName(event.target.value)}
                  placeholder="mistral-small3.2:latest"
                  className={css({
                    w: "full",
                    px: 3,
                    py: 2.5,
                    rounded: "md",
                    bg: "gray.900",
                    border: "1px solid",
                    borderColor: "gray.700",
                    color: "white",
                    _focus: {
                      outline: "none",
                      borderColor: "blue.500",
                    },
                  })}
                />
                <button
                  onClick={() => void handleCustomPull()}
                  disabled={busyModelName !== null}
                  className={css({
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 2,
                    px: 4,
                    py: 2.5,
                    rounded: "md",
                    bg: "blue.600",
                    color: "white",
                    _hover: { bg: "blue.700" },
                    _disabled: { opacity: 0.5, cursor: "not-allowed" },
                  })}
                >
                  <Download size={16} />
                  {busyModelName === customModelName.trim() && customModelName.trim()
                    ? "Pulling..."
                    : "Pull Custom Model"}
                </button>
              </div>

              <div className={css({ display: "grid", gap: 2 })}>
                <h3 className={css({ fontSize: "sm", color: "gray.300" })}>
                  Supported Ollama models
                </h3>
                {availableSupportedModels.length === 0 ? (
                  <div
                    className={css({
                      px: 3,
                      py: 3,
                      rounded: "md",
                      bg: "gray.900",
                      color: "gray.400",
                      border: "1px solid",
                      borderColor: "gray.700",
                      fontSize: "sm",
                    })}
                  >
                    All supported Ollama models are already installed.
                  </div>
                ) : (
                  availableSupportedModels.map((modelName) => {
                    const isBusy = busyModelName === modelName;
                    return (
                      <button
                        key={modelName}
                        onClick={() => void handlePull(modelName)}
                        disabled={busyModelName !== null}
                        className={css({
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                          gap: 3,
                          px: 3,
                          py: 3,
                          rounded: "md",
                          bg: "gray.900",
                          color: "gray.200",
                          border: "1px solid",
                          borderColor: "gray.700",
                          textAlign: "left",
                          _hover: { bg: "gray.800", borderColor: "gray.600" },
                          _disabled: { opacity: 0.5, cursor: "not-allowed" },
                        })}
                      >
                        <span className={css({ wordBreak: "break-word" })}>
                          {modelName}
                        </span>
                        <span className={css({ color: "blue.300", whiteSpace: "nowrap" })}>
                          {isBusy ? "Pulling..." : "Pull"}
                        </span>
                      </button>
                    );
                  })
                )}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
