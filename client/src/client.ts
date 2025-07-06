import { Message } from "./types";

interface ClientConfig {
  host?: string;
  port?: number;
}

export class AgentClient {
  private host: string;
  private port: number;

  constructor({ host = "localhost", port = 8000 }: ClientConfig = {}) {
    this.host = host;
    this.port = port;
  }

  get httpBaseUrl(): string {
    return `http://${this.host}:${this.port}`;
  }

  get wsBaseUrl(): string {
    return `ws://${this.host}:${this.port}`;
  }

  get chatWsUrl(): string {
    return `${this.wsBaseUrl}/api/chat`;
  }

  get resetUrl(): string {
    return `${this.httpBaseUrl}/api/reset`;
  }

  async reset(): Promise<void> {
    const response = await fetch(this.resetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Reset failed: ${response.statusText}`);
    }
  }

  async getConversation(): Promise<Message[]> {
    const response = await fetch(`${this.httpBaseUrl}/api/conversation`);

    if (!response.ok) {
      throw new Error(`Failed to get conversation: ${response.statusText}`);
    }

    const data = await response.json();
    return data.messages;
  }

  async getConfig(): Promise<{
    name: string;
    description: string;
    tools: string[];
  }> {
    const response = await fetch(`${this.httpBaseUrl}/api/config`);

    if (!response.ok) {
      throw new Error(`Failed to get config: ${response.statusText}`);
    }

    return response.json();
  }

  async getState(): Promise<Record<string, any>> {
    const response = await fetch(`${this.httpBaseUrl}/api/state`);

    if (!response.ok) {
      throw new Error(`Failed to get state: ${response.statusText}`);
    }

    return response.json();
  }

  async getContextInfo(): Promise<{
    message_count: number;
    conversation_messages: number;
    estimated_tokens: number;
    context_limit: number;
    usage_percentage: number;
    approaching_limit: boolean;
  }> {
    const response = await fetch(`${this.httpBaseUrl}/api/context`);

    if (!response.ok) {
      throw new Error(`Failed to get context info: ${response.statusText}`);
    }

    return response.json();
  }
}
