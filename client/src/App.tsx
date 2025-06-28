import { useMemo } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { AgentClient } from './client';

function App() {
  const client = useMemo(() => new AgentClient(), []);
  
  return <ChatInterface client={client} />;
}

export default App;