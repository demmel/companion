import { renderHook, act } from "@testing-library/react";
import { useConversation } from "../useConversation";
import { UserMessage, AgentMessage } from "../../types";
import { ClientAgentEvent } from "../useWebSocket";

describe("useConversation", () => {
  it("should start with empty messages", () => {
    const { result } = renderHook(() => useConversation([]));

    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it("should add user messages", () => {
    const { result } = renderHook(() => useConversation([]));

    act(() => {
      result.current.addUserMessage("Hello");
    });

    expect(result.current.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "Hello" }] },
    ]);
  });

  it("should show streaming agent response as it comes", () => {
    const events: ClientAgentEvent[] = [
      { id: 0, type: "text", content: "Hello " },
      { id: 1, type: "text", content: "there!" },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello there!" }],
        tool_calls: [],
      },
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it("should handle tool calls during streaming", () => {
    const events: ClientAgentEvent[] = [
      { id: 0, type: "text", content: "Let me help you. " },
      {
        id: 1,
        type: "tool_started",
        tool_name: "set_mood",
        tool_id: "call_1",
        parameters: { mood: "happy" },
      },
      {
        id: 2,
        type: "tool_finished",
        tool_id: "call_1",
        result: {
          type: "success",
          content: { type: "text", text: "Mood set to happy" },
        },
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Let me help you. " }],
        tool_calls: [
          {
            type: "finished",
            tool_name: "set_mood",
            tool_id: "call_1",
            parameters: { mood: "happy" },
            result: {
              type: "success",
              content: "Mood set to happy",
            },
          },
        ],
      },
    ]);
  });

  it("should show running tool calls before they finish", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "tool_started",
        tool_name: "slow_tool",
        tool_id: "call_1",
        parameters: { task: "processing" },
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [],
        tool_calls: [
          {
            type: "started",
            tool_name: "slow_tool",
            tool_id: "call_1",
            parameters: { task: "processing" },
          },
        ],
      },
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it("should finalize message on response_complete", () => {
    let events: ClientAgentEvent[] = [
      { id: 0, type: "text", content: "Done!" },
    ];

    const { result, rerender } = renderHook(
      (props) => useConversation(props.events),
      {
        initialProps: { events },
      },
    );

    // Should be streaming
    expect(result.current.isStreamActive).toBe(true);

    // Add response_complete event
    events = [
      ...events,
      {
        id: 1,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    rerender({ events });

    expect(result.current.isStreamActive).toBe(false);
    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Done!" }],
        tool_calls: [],
      },
    ]);
  });

  it("should load conversation from API", () => {
    const { result } = renderHook(() => useConversation([]));

    const conversationMessages: [UserMessage, AgentMessage] = [
      { role: "user", content: [{ type: "text", text: "Hello" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "Hi there!" }],
        tool_calls: [],
      },
    ];

    act(() => {
      result.current.loadConversation(conversationMessages);
    });

    expect(result.current.messages).toEqual(conversationMessages);
  });

  it("should clear conversation", () => {
    const { result } = renderHook(() => useConversation([]));

    act(() => {
      result.current.addUserMessage("Hello");
      result.current.clearConversation();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it("should handle multiple tool calls", () => {
    const events: ClientAgentEvent[] = [
      { id: 0, type: "text", content: "Working on it. " },
      {
        id: 1,
        type: "tool_started",
        tool_name: "tool_a",
        tool_id: "call_1",
        parameters: { a: 1 },
      },
      {
        id: 2,
        type: "tool_started",
        tool_name: "tool_b",
        tool_id: "call_2",
        parameters: { b: 2 },
      },
      {
        id: 3,
        type: "tool_finished",
        tool_id: "call_1",
        result: {
          type: "success",
          content: { type: "text", text: "Result A" },
        },
      },
      {
        id: 4,
        type: "tool_finished",
        tool_id: "call_2",
        result: { type: "error", error: "Error B" },
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    const message = result.current.messages[0] as AgentMessage;
    expect(message.tool_calls).toHaveLength(2);

    const toolA = message.tool_calls.find((t) => t.tool_id === "call_1");
    const toolB = message.tool_calls.find((t) => t.tool_id === "call_2");

    expect(toolA).toEqual({
      type: "finished",
      tool_name: "tool_a",
      tool_id: "call_1",
      parameters: { a: 1 },
      result: { type: "success", content: "Result A" },
    });

    expect(toolB).toEqual({
      type: "finished",
      tool_name: "tool_b",
      tool_id: "call_2",
      parameters: { b: 2 },
      result: { type: "error", content: "Error B" },
    });
  });

  it("should not duplicate messages", () => {
    let events: ClientAgentEvent[] = [
      { id: 0, type: "text", content: "Hello" },
      { id: 1, type: "text", content: " World" },
    ];

    const { result, rerender } = renderHook(
      ({ events }) => useConversation(events),
      {
        initialProps: { events },
      },
    );

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World" }],
        tool_calls: [],
      },
    ]);

    // Add another event that should not duplicate
    events = [
      ...events,
      { id: 2, type: "text", content: "!" },
      {
        id: 3,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World!" }],
        tool_calls: [],
      },
    ]);

    // Add a new user message
    act(() => {
      result.current.addUserMessage("New message");
    });

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World!" }],
        tool_calls: [],
      },
      { role: "user", content: [{ type: "text", text: "New message" }] },
    ]);

    // Add another agent message
    events = [
      ...events,
      { id: 4, type: "text", content: "Another response" },
      {
        id: 5,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World!" }],
        tool_calls: [],
      },
      { role: "user", content: [{ type: "text", text: "New message" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "Another response" }],
        tool_calls: [],
      },
    ]);

    // Add a new user message again
    act(() => {
      result.current.addUserMessage("Another user message");
    });

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World!" }],
        tool_calls: [],
      },
      { role: "user", content: [{ type: "text", text: "New message" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "Another response" }],
        tool_calls: [],
      },
      {
        role: "user",
        content: [{ type: "text", text: "Another user message" }],
      },
    ]);

    // Add tool calls
    events = [
      ...events,
      {
        id: 6,
        type: "tool_started",
        tool_name: "example_tool",
        tool_id: "call_3",
        parameters: { param: "value" },
      },
      {
        id: 7,
        type: "tool_finished",
        tool_id: "call_3",
        result: {
          type: "success",
          content: { type: "text", text: "Tool call result" },
        },
      },
      { id: 8, type: "text", content: "Final response" },
      {
        id: 9,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    rerender({ events });

    expect(result.current.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello World!" }],
        tool_calls: [],
      },
      { role: "user", content: [{ type: "text", text: "New message" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "Another response" }],
        tool_calls: [],
      },
      {
        role: "user",
        content: [{ type: "text", text: "Another user message" }],
      },
      {
        role: "assistant",
        content: [],
        tool_calls: [
          {
            type: "finished",
            tool_name: "example_tool",
            tool_id: "call_3",
            parameters: { param: "value" },
            result: { type: "success", content: "Tool call result" },
          },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "Final response" }],
        tool_calls: [],
      },
    ]);
  });

  it("should handle summarization events correctly", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 10,
        recent_messages_kept: 6,
        context_usage_before: 85.5,
      },
      {
        id: 1,
        type: "summarization_finished",
        summary:
          "User and assistant discussed various topics including weather, food preferences, and travel plans.",
        messages_summarized: 10,
        messages_after: 7,
        context_usage_after: 42.3,
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("system");

    const content = result.current.messages[0].content[0] as any;
    expect(content.type).toBe("summarization");
    expect(content.title).toContain("✅ Summarized 10 messages");
    expect(content.summary).toBe(
      "User and assistant discussed various topics including weather, food preferences, and travel plans.",
    );
    expect(content.messages_summarized).toBe(10);
    expect((content as any).context_usage_before).toBe(85.5);
    expect((content as any).context_usage_after).toBe(42.3);
  });

  it("should show summarization progress during streaming", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 8,
        recent_messages_kept: 4,
        context_usage_before: 78.2,
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("system");
    expect(result.current.messages[0].content).toEqual([
      {
        type: "text",
        text: "📝 Summarizing 8 older messages to manage context (78.2% usage)...",
      },
    ]);
    expect(result.current.isStreamActive).toBe(true);
  });

  it("should complete summarization and stop streaming", () => {
    let events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 5,
        recent_messages_kept: 3,
        context_usage_before: 90.1,
      },
    ];

    const { result, rerender } = renderHook(
      ({ events }) => useConversation(events),
      {
        initialProps: { events },
      },
    );

    expect(result.current.isStreamActive).toBe(true);

    // Complete the summarization
    events = [
      ...events,
      {
        id: 1,
        type: "summarization_finished",
        summary: "Previous conversation about project planning and deadlines.",
        messages_summarized: 5,
        messages_after: 4,
        context_usage_after: 35.7,
      },
    ];

    rerender({ events });

    expect(result.current.isStreamActive).toBe(true); // Still streaming until response_complete
    expect(result.current.messages).toHaveLength(1);

    const content = result.current.messages[0].content[0] as any;
    expect(content.type).toBe("summarization");
    expect(content.context_usage_before).toBe(90.1);
    expect(content.context_usage_after).toBe(35.7);
  });

  it("should handle summarization followed by regular response", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 6,
        recent_messages_kept: 4,
        context_usage_before: 82.0,
      },
      {
        id: 1,
        type: "summarization_finished",
        summary: "Discussion about travel destinations and budget planning.",
        messages_summarized: 6,
        messages_after: 5,
        context_usage_after: 38.5,
      },
      { id: 2, type: "text", content: "Now I can continue helping you!" },
      {
        id: 3,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(2);

    // First message: summarization
    expect(result.current.messages[0].role).toBe("system");
    expect((result.current.messages[0].content[0] as any).type).toBe(
      "summarization",
    );

    // Second message: regular assistant response
    expect(result.current.messages[1].role).toBe("assistant");
    expect(result.current.messages[1].content).toEqual([
      { type: "text", text: "Now I can continue helping you!" },
    ]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it("should maintain correct role for system messages through response_complete", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 5,
        recent_messages_kept: 3,
        context_usage_before: 85.0,
      },
      {
        id: 1,
        type: "summarization_finished",
        summary: "Test summary content",
        messages_summarized: 5,
        messages_after: 4,
        context_usage_after: 40.0,
      },
      {
        id: 2,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("system"); // This should NOT be 'assistant'
    expect(result.current.messages[0]).not.toHaveProperty("tool_calls");
    expect(result.current.isStreamActive).toBe(false);
  });

  it("should not create AgentMessage with SummarizationContent when text events follow summarization", () => {
    // This test covers the specific bug where text events after summarization_finished
    // would reset role to 'assistant' but leave SummarizationContent as content,
    // causing agentMessage.content.trim() to fail in RoleplayPresenter
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "summarization_started",
        messages_to_summarize: 6,
        recent_messages_kept: 4,
        context_usage_before: 75.0,
      },
      {
        id: 1,
        type: "summarization_finished",
        summary: "Previous conversation about exploring nature and mushrooms.",
        messages_summarized: 6,
        messages_after: 8,
        context_usage_after: 68.0,
      },
      // Text events that follow summarization (this was causing the bug)
      { id: 2, type: "text", content: "*Eyes light up*" },
      { id: 3, type: "text", content: " Amazing facts!" },
      {
        id: 4,
        type: "response_complete",
        message_count: 1,
        conversation_messages: 1,
        estimated_tokens: 10,
        context_limit: 1000,
        usage_percentage: 1.0,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useConversation(events));

    expect(result.current.messages).toHaveLength(2);

    // First message: properly created system message with SummarizationContent
    const systemMessage = result.current.messages[0];
    expect(systemMessage.role).toBe("system");
    const content = systemMessage.content[0] as any;
    expect(content.type).toBe("summarization");
    expect(content.summary).toBe(
      "Previous conversation about exploring nature and mushrooms.",
    );
    expect(systemMessage).not.toHaveProperty("tool_calls");

    // Second message: properly created assistant message with string content
    const assistantMessage = result.current.messages[1];
    expect(assistantMessage.role).toBe("assistant");
    expect(assistantMessage.content).toEqual([
      { type: "text", text: "*Eyes light up* Amazing facts!" },
    ]);
    expect(assistantMessage).toHaveProperty("tool_calls");
    expect((assistantMessage as any).tool_calls).toEqual([]);

    expect(result.current.isStreamActive).toBe(false);
  });
});
