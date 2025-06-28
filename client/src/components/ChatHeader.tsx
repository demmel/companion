import { Wifi, WifiOff } from 'lucide-react';

interface ChatHeaderProps {
  title?: string;
  isConnected?: boolean;
  isConnecting?: boolean;
}

export function ChatHeader({ 
  title = "Agent Chat", 
  isConnected = false, 
  isConnecting = false 
}: ChatHeaderProps) {
  const getConnectionStatus = () => {
    if (isConnecting) return { icon: Wifi, text: 'Connecting...', color: 'text-yellow-500' };
    if (isConnected) return { icon: Wifi, text: 'Connected', color: 'text-green-500' };
    return { icon: WifiOff, text: 'Disconnected', color: 'text-red-500' };
  };

  const status = getConnectionStatus();
  const StatusIcon = status.icon;

  return (
    <div className="bg-white border-b px-4 py-3 flex items-center justify-between">
      <h1 className="text-lg font-semibold text-gray-900">{title}</h1>
      <div className="flex items-center gap-2">
        <StatusIcon size={16} className={status.color} />
        <span className={`text-sm ${status.color}`}>{status.text}</span>
      </div>
    </div>
  );
}