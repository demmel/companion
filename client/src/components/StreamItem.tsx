import { RoleplayText } from './RoleplayText';
import { Wrench, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface StreamItemProps {
  data: 
    | { type: 'text'; content: string; role: 'user' | 'agent' }
    | { 
        type: 'tool'; 
        toolId: string; 
        name: string; 
        parameters: Record<string, any>; 
        status: 'running' | 'completed' | 'error';
        result?: string;
        error?: string;
      };
}

function ToolDisplay({ data }: { data: Extract<StreamItemProps['data'], { type: 'tool' }> }) {
  const getStatusIcon = () => {
    switch (data.status) {
      case 'running':
        return <Loader2 size={16} className="animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'error':
        return <XCircle size={16} className="text-red-500" />;
    }
  };

  const getStatusColor = () => {
    switch (data.status) {
      case 'running':
        return 'border-blue-200 bg-blue-50';
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
    }
  };

  return (
    <div className={`border rounded-lg p-3 my-2 ${getStatusColor()}`}>
      <div className="flex items-center gap-2 mb-2">
        <Wrench size={14} className="text-gray-500" />
        <span className="text-sm font-medium">{data.name}</span>
        {getStatusIcon()}
      </div>
      
      {Object.keys(data.parameters).length > 0 && (
        <div className="text-xs text-gray-600 mb-2">
          <details className="cursor-pointer">
            <summary className="font-medium">Parameters</summary>
            <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto mt-1">
              {JSON.stringify(data.parameters, null, 2)}
            </pre>
          </details>
        </div>
      )}
      
      {data.result && (
        <div className="text-sm text-gray-700">
          <div className="font-medium text-xs text-gray-500 mb-1">Result:</div>
          <div className="bg-gray-100 p-2 rounded text-xs">
            {data.result}
          </div>
        </div>
      )}
      
      {data.error && (
        <div className="text-sm text-red-700">
          <div className="font-medium text-xs text-red-500 mb-1">Error:</div>
          <div className="bg-red-50 p-2 rounded text-xs">
            {data.error}
          </div>
        </div>
      )}
    </div>
  );
}

export function StreamItem({ data }: StreamItemProps) {
  if (data.type === 'text') {
    return <RoleplayText content={data.content} />;
  }
  
  return <ToolDisplay data={data} />;
}