import {
  ToolCall,
  ToolCallFinished,
  ImageGenerationToolContent,
} from "../types";
import { Wrench, CheckCircle, XCircle, Loader2 } from "lucide-react";

interface ToolDisplayProps {
  toolCall: ToolCall;
}

export function ToolDisplay({ toolCall }: ToolDisplayProps) {
  const renderToolResult = (toolCall: ToolCallFinished) => {
    if (toolCall.result.type === "error") {
      return (
        <div className="p-2 rounded text-xs bg-red-50 text-red-700">
          {toolCall.result.error}
        </div>
      );
    }

    // Handle success case with typed content
    const content = toolCall.result.content;

    switch (content.type) {
      case "image_generated":
        return (
          <div className="space-y-2">
            <img
              src={content.image_url}
              alt={content.prompt}
              className="max-w-full h-auto rounded-lg border"
              style={{ maxHeight: "300px" }}
            />
            <div className="text-xs text-gray-600 space-y-1">
              <div>
                <strong>Prompt:</strong> {content.prompt}
              </div>
              <div>
                <strong>Size:</strong> {content.width}×{content.height}
              </div>
              <div>
                <strong>Steps:</strong> {content.num_inference_steps}
              </div>
              {content.negative_prompt && (
                <div>
                  <strong>Negative:</strong> {content.negative_prompt}
                </div>
              )}
              {content.seed && (
                <div>
                  <strong>Seed:</strong> {content.seed}
                </div>
              )}
            </div>
          </div>
        );

      case "text":
        return (
          <div className="p-2 rounded text-xs bg-gray-100">{content.text}</div>
        );

      default:
        // Fallback for unknown content types
        return (
          <div className="p-2 rounded text-xs bg-gray-100">
            <pre className="overflow-x-auto">
              {JSON.stringify(content, null, 2)}
            </pre>
          </div>
        );
    }
  };

  const getStatusIcon = () => {
    if (toolCall.type === "started") {
      return <Loader2 size={16} className="animate-spin text-blue-500" />;
    }

    const finished = toolCall as ToolCallFinished;
    switch (finished.result.type) {
      case "success":
        return <CheckCircle size={16} className="text-green-500" />;
      case "error":
        return <XCircle size={16} className="text-red-500" />;
    }
  };

  const getStatusColor = () => {
    if (toolCall.type === "started") {
      return "border-blue-200 bg-blue-50";
    }

    const finished = toolCall as ToolCallFinished;
    switch (finished.result.type) {
      case "success":
        return "border-green-200 bg-green-50";
      case "error":
        return "border-red-200 bg-red-50";
    }
  };

  return (
    <div className={`border rounded-lg p-3 my-2 ${getStatusColor()}`}>
      <div className="flex items-center gap-2 mb-2">
        <Wrench size={14} className="text-gray-500" />
        <span className="text-sm font-medium">{toolCall.tool_name}</span>
        {getStatusIcon()}
        <span className="text-xs text-gray-500">({toolCall.tool_id})</span>
      </div>

      {Object.keys(toolCall.parameters).length > 0 && (
        <div className="text-xs text-gray-600 mb-2">
          <details className="cursor-pointer">
            <summary className="font-medium">Parameters</summary>
            <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto mt-1">
              {JSON.stringify(toolCall.parameters, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {toolCall.type === "finished" && (
        <div className="text-sm text-gray-700">
          <div className="font-medium text-xs text-gray-500 mb-1">Result:</div>
          {renderToolResult(toolCall)}
        </div>
      )}
    </div>
  );
}
