import { Message, AgentMessage } from '../types';
import { RoleplayText } from './RoleplayText';
import { ToolDisplay } from './ToolDisplay';

interface MessageItemProps {
  message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
  if (message.role === 'user') {
    return (
      <div className="mb-4">
        <div className="text-sm text-gray-600 mb-1">You</div>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    );
  }

  // Agent message
  const agentMessage = message as AgentMessage;
  
  return (
    <div className="mb-4">
      <div className="text-sm text-gray-600 mb-1">Agent</div>
      
      {/* Message content */}
      {agentMessage.content && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-2">
          <RoleplayText content={agentMessage.content} />
        </div>
      )}
      
      {/* Tool calls */}
      {agentMessage.tool_calls.map((toolCall, index) => (
        <div key={`${toolCall.tool_id}-${index}`} className="mt-2">
          <ToolDisplay toolCall={toolCall} />
        </div>
      ))}
    </div>
  );
}