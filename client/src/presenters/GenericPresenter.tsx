import { ConversationPresenterProps } from "./types";
import { Message, AgentMessage } from "../types";
import { RoleplayText } from "../components/RoleplayText";
import { ToolDisplay } from "../components/ToolDisplay";

export function GenericPresenter({
  messages,
  isStreamActive,
}: ConversationPresenterProps) {
  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <MessageItem key={index} message={message} />
      ))}

      {/* Show streaming cursor when response is active */}
      {isStreamActive && messages.length > 0 && (
        <div className="prose prose-sm max-w-none">
          <span className="animate-pulse text-gray-500">▋</span>
        </div>
      )}
    </div>
  );
}

function MessageItem({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="mb-4">
        <div className="text-sm text-gray-600 mb-1">You</div>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="whitespace-pre-wrap">
            {message.content.map((item, index) => {
              if (item.type === "text") {
                return <span key={index}>{item.text}</span>;
              }
              // Handle other content types if needed
              return null;
            })}
          </div>
        </div>
      </div>
    );
  }

  // Agent message
  const agentMessage = message as AgentMessage;

  return (
    <div className="mb-4">
      <div className="text-sm text-gray-600 mb-1">🤖 Agent</div>

      {/* Message content */}
      {agentMessage.content && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-2">
          {agentMessage.content.map((item, index) => (
            <RoleplayText key={index} content={item.text} />
          ))}
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
