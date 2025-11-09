import { useState, useEffect } from "react";
import { X, Cpu } from "lucide-react";
import { css } from "@styled-system/css";
import { AgentClient } from "@/client";
import { ModelConfigResponse } from "@/types";

interface ModelSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  client: AgentClient;
}

const MODEL_LABELS: Record<string, string> = {
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
};

export function ModelSettings({ isOpen, onClose, client }: ModelSettingsProps) {
  const [config, setConfig] = useState<ModelConfigResponse | null>(null);
  const [supportedModels, setSupportedModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [modelConfig, modelsResponse] = await Promise.all([
        client.getModelConfig(),
        client.getSupportedModels(),
      ]);
      setConfig(modelConfig);
      setSupportedModels(modelsResponse.models);
    } catch (err) {
      setError(`Failed to load configuration: ${err}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!config) return;

    setIsSaving(true);
    setError(null);
    try {
      await client.updateModelConfig(config);
      onClose();
    } catch (err) {
      setError(`Failed to save model configuration: ${err}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleModelChange = (field: keyof ModelConfigResponse, value: string) => {
    if (!config) return;
    setConfig({
      ...config,
      [field]: value,
    });
  };

  if (!isOpen) return null;

  return (
    <div
      className={css({
        position: "absolute",
        top: "100%",
        left: 0,
        mt: 2,
        bg: "gray.800",
        border: "1px solid",
        borderColor: "gray.600",
        rounded: "lg",
        width: "360px",
        maxHeight: "500px",
        display: "flex",
        flexDirection: "column",
        zIndex: 50,
        boxShadow: "lg",
      })}
    >
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          p: 3,
          borderBottom: "1px solid",
          borderColor: "gray.600",
          flexShrink: 0,
        })}
      >
        <div
          className={css({
            display: "flex",
            alignItems: "center",
            gap: 2,
          })}
        >
          <Cpu size={16} className={css({ color: "gray.400" })} />
          <h3
            className={css({
              fontSize: "sm",
              fontWeight: "medium",
              color: "white",
            })}
          >
            Model Configuration
          </h3>
        </div>
        <button
          onClick={onClose}
          className={css({
            p: 1,
            color: "gray.400",
            _hover: { color: "white" },
            transition: "colors",
            rounded: "md",
          })}
        >
          <X size={14} />
        </button>
      </div>

      <div
        className={css({
          overflowY: "auto",
          flex: 1,
          p: 3,
        })}
      >
        {error && (
          <div
            className={css({
              mb: 3,
              p: 2,
              bg: "red.900",
              border: "1px solid",
              borderColor: "red.600",
              rounded: "md",
              fontSize: "xs",
              color: "red.200",
            })}
          >
            {error}
          </div>
        )}

        {isLoading ? (
          <div className={css({ textAlign: "center", py: 4, color: "gray.400" })}>
            Loading configuration...
          </div>
        ) : config ? (
          <div className={css({ display: "flex", flexDirection: "column", gap: 2.5 })}>
            {(Object.keys(MODEL_LABELS) as Array<keyof ModelConfigResponse>).map(
              (field) => (
                <div key={field}>
                  <label
                    className={css({
                      display: "block",
                      fontSize: "xs",
                      fontWeight: "medium",
                      color: "gray.300",
                      mb: 1,
                    })}
                  >
                    {MODEL_LABELS[field]}
                  </label>
                  <select
                    value={config[field]}
                    onChange={(e) => handleModelChange(field, e.target.value)}
                    className={css({
                      w: "full",
                      px: 2,
                      py: 1.5,
                      bg: "gray.700",
                      border: "1px solid",
                      borderColor: "gray.600",
                      rounded: "md",
                      color: "white",
                      fontSize: "xs",
                      _focus: {
                        outline: "none",
                        borderColor: "blue.500",
                      },
                    })}
                  >
                    {supportedModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
              )
            )}
          </div>
        ) : null}
      </div>

      {config && (
        <>
          <div
            className={css({
              p: 3,
              borderTop: "1px solid",
              borderColor: "gray.600",
              flexShrink: 0,
            })}
          >
            <div className={css({ display: "flex", gap: 2, mb: 2 })}>
              <button
                onClick={handleSave}
                disabled={isSaving}
                className={css({
                  flex: 1,
                  px: 3,
                  py: 2,
                  bg: "blue.600",
                  color: "white",
                  rounded: "md",
                  fontSize: "sm",
                  fontWeight: "medium",
                  _hover: { bg: "blue.700" },
                  _disabled: {
                    bg: "gray.600",
                    cursor: "not-allowed",
                  },
                  transition: "colors",
                })}
              >
                {isSaving ? "Saving..." : "Save"}
              </button>
              <button
                onClick={onClose}
                disabled={isSaving}
                className={css({
                  px: 3,
                  py: 2,
                  bg: "gray.700",
                  color: "gray.300",
                  rounded: "md",
                  fontSize: "sm",
                  fontWeight: "medium",
                  _hover: { bg: "gray.600" },
                  _disabled: {
                    cursor: "not-allowed",
                  },
                  transition: "colors",
                })}
              >
                Cancel
              </button>
            </div>

            <p
              className={css({
                fontSize: "xs",
                color: "gray.500",
              })}
            >
              Changes take effect on the next user message.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
