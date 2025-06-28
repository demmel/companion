import { Message } from '../types';

export interface ConversationPresenterProps {
  messages: Message[];
  isStreamActive: boolean;
  agentState?: Record<string, any>; // Agent state for character tracking
}

export type ConversationPresenter = React.ComponentType<ConversationPresenterProps>;