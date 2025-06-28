import { ToolCall, ToolCallFinished } from '../types';
import { Wrench, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface ToolDisplayProps {
  toolCall: ToolCall;
}

export function ToolDisplay({ toolCall }: ToolDisplayProps) {
  const getStatusIcon = () => {
    if (toolCall.type === 'started') {
      return <Loader2 size={16} className="animate-spin text-blue-500" />;
    }
    
    const finished = toolCall as ToolCallFinished;
    switch (finished.result.type) {
      case 'success':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'error':
        return <XCircle size={16} className="text-red-500" />;
    }
  };

  const getStatusColor = () => {
    if (toolCall.type === 'started') {
      return 'border-blue-200 bg-blue-50';
    }
    
    const finished = toolCall as ToolCallFinished;
    switch (finished.result.type) {
      case 'success':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
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
      
      {toolCall.type === 'finished' && (
        <div className="text-sm text-gray-700">
          <div className="font-medium text-xs text-gray-500 mb-1">Result:</div>
          <div className={`p-2 rounded text-xs ${
            toolCall.result.type === 'error' 
              ? 'bg-red-50 text-red-700' 
              : 'bg-gray-100'
          }`}>
            {toolCall.result.content}
          </div>
        </div>
      )}
    </div>
  );
}