import { render, screen } from "@testing-library/react";
import { RoleplayPresenter } from "../RoleplayPresenter";
import {
  UserMessage,
  AgentMessage,
  ToolCallFinished,
  Message,
} from "../../types";

describe("RoleplayPresenter", () => {
  const mockAgentState = {
    current_character_id: null,
    characters: {},
    global_scene: null,
    global_memories: [],
  };

  it("should show generic agent header when no character is active", () => {
    const messages: UserMessage[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Hello" }],
      },
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("should hide assume_character tool and not show header until character speaks", () => {
    const assumeCharacterTool: ToolCallFinished = {
      type: "finished",
      tool_name: "assume_character",
      tool_id: "call_1",
      parameters: {
        character_name: "Bob",
        personality: "friendly and curious",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Character created" },
      },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Can you play as Bob?" }],
      },
      {
        role: "assistant",
        content: [],
        tool_calls: [assumeCharacterTool],
      } as AgentMessage,
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Say hello" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Hello there!" }],
        tool_calls: [],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should NOT show the tool call
    expect(screen.queryByText("assume_character")).not.toBeInTheDocument();

    // Should show character header when they speak
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("😐")).toBeInTheDocument(); // neutral mood emoji

    // Should show dialogue
    expect(screen.getByText("Hello there!")).toBeInTheDocument();
  });

  it("should track character state evolution over conversation", () => {
    const assumeTool: ToolCallFinished = {
      type: "finished",
      tool_name: "assume_character",
      tool_id: "call_1",
      parameters: {
        character_name: "Alice",
        personality: "mysterious",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Character created" },
      },
    };

    const moodTool: ToolCallFinished = {
      type: "finished",
      tool_name: "set_mood",
      tool_id: "call_2",
      parameters: {
        mood: "happy",
        intensity: "high",
      },
      result: { type: "success", content: { type: "text", text: "Mood set" } },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Play as Alice" }],
      },
      {
        role: "assistant",
        content: [],
        tool_calls: [assumeTool],
      } as AgentMessage,
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Be happy" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "I feel great!" }],
        tool_calls: [moodTool],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should show Alice in the character header
    expect(screen.getByText("Alice")).toBeInTheDocument();

    // Should show happy mood in final message (may have multiple instances)
    expect(screen.getAllByText("😊")).toHaveLength(2); // happy emoji in header + transition
    expect(screen.getByText(/happy.*high/)).toBeInTheDocument(); // Match mood format with bullet points

    // Should show the dialogue content
    expect(screen.getByText("I feel great!")).toBeInTheDocument();
  });

  it("should hide memory tools", () => {
    const hiddenTool: ToolCallFinished = {
      type: "finished",
      tool_name: "remember_detail",
      tool_id: "call_1",
      parameters: {
        detail: "User likes coffee",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Memory stored" },
      },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "I like coffee" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Got it!" }],
        tool_calls: [hiddenTool],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should NOT show hidden tool
    expect(screen.queryByText("remember_detail")).not.toBeInTheDocument();
    expect(screen.queryByText("Memory stored")).not.toBeInTheDocument();

    // Should show dialogue
    expect(screen.getByText("Got it!")).toBeInTheDocument();
  });

  it("should show special presentations for roleplay tools", () => {
    const actionTool: ToolCallFinished = {
      type: "finished",
      tool_name: "character_action",
      tool_id: "call_1",
      parameters: {
        action: "waves enthusiastically",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Action performed" },
      },
    };

    const thoughtTool: ToolCallFinished = {
      type: "finished",
      tool_name: "internal_thought",
      tool_id: "call_2",
      parameters: {
        thought: "This person seems nice",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Thought recorded" },
      },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Hello" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Hi there!" }],
        tool_calls: [actionTool, thoughtTool],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should show action in italics with asterisks
    expect(screen.getByText("*waves enthusiastically*")).toBeInTheDocument();

    // Should show thought components (text is split across spans)
    expect(screen.getByText("💭")).toBeInTheDocument();
    expect(screen.getByText("This person seems nice")).toBeInTheDocument();

    // Should show dialogue
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should only show character header when character changes", () => {
    const assumeBob: ToolCallFinished = {
      type: "finished",
      tool_name: "assume_character",
      tool_id: "call_1",
      parameters: { character_name: "Bob", personality: "friendly" },
      result: {
        type: "success",
        content: { type: "text", text: "Character created" },
      },
    };

    const assumeAlice: ToolCallFinished = {
      type: "finished",
      tool_name: "assume_character",
      tool_id: "call_2",
      parameters: { character_name: "Alice", personality: "mysterious" },
      result: {
        type: "success",
        content: { type: "text", text: "Character created" },
      },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Play as Bob" }],
      },
      {
        role: "assistant",
        content: [],
        tool_calls: [assumeBob],
      } as AgentMessage,
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Say hi" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Hello!" }],
        tool_calls: [],
      } as AgentMessage,
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Now play as Alice" }],
      },
      {
        role: "assistant",
        content: [],
        tool_calls: [assumeAlice],
      } as AgentMessage,
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Say something mysterious" }],
      },
      {
        role: "assistant",
        content: [
          { type: "text" as const, text: "The shadows whisper secrets..." },
        ],
        tool_calls: [],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should show both character names (headers only appear when character changes)
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();

    // Should show both dialogues
    expect(screen.getByText("Hello!")).toBeInTheDocument();
    expect(
      screen.getByText("The shadows whisper secrets..."),
    ).toBeInTheDocument();
  });

  it("should show scene setting", () => {
    const sceneTool: ToolCallFinished = {
      type: "finished",
      tool_name: "scene_setting",
      tool_id: "call_1",
      parameters: {
        location: "Dark alley",
        atmosphere: "mysterious",
        time: "midnight",
      },
      result: { type: "success", content: { type: "text", text: "Scene set" } },
    };

    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Set the scene" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "The scene is set." }],
        tool_calls: [sceneTool],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    // Should show scene setting
    // Check for scene setting components (text may be split across spans)
    expect(screen.getByText("📍")).toBeInTheDocument();
    expect(
      screen.getByText(/Dark alley • mysterious • \(midnight\)/),
    ).toBeInTheDocument();

    // Should show dialogue
    expect(screen.getByText("The scene is set.")).toBeInTheDocument();
  });

  it("should show streaming cursor when active", () => {
    const messages: Message[] = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Hello" }],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Hi there!" }],
        tool_calls: [],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={true}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("▋")).toBeInTheDocument();
  });

  it("should handle system messages with text content", () => {
    const messages = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Hello" }],
      },
      {
        role: "system" as const,
        content: [
          {
            type: "text" as const,
            text: "System notification: Context updated",
          },
        ],
      },
      {
        role: "assistant" as const,
        content: [{ type: "text" as const, text: "How can I help?" }],
        tool_calls: [],
      },
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(
      screen.getByText("System notification: Context updated"),
    ).toBeInTheDocument();
    expect(screen.getByText("How can I help?")).toBeInTheDocument();
  });

  it("should handle system messages with structured content", () => {
    const summarizationContent = {
      type: "summarization" as const,
      title: "✅ Summarized 5 messages. Context usage: 80% → 40%",
      summary: "Previous conversation about weather and travel",
      messages_summarized: 5,
      context_usage_before: 80,
      context_usage_after: 40,
    };

    const messages = [
      {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Can you help me?" }],
      },
      { role: "system" as const, content: [summarizationContent] },
      {
        role: "assistant" as const,
        content: [{ type: "text" as const, text: "Of course!" }],
        tool_calls: [],
      },
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("Can you help me?")).toBeInTheDocument();
    expect(
      screen.getByText("✅ Summarized 5 messages. Context usage: 80% → 40%"),
    ).toBeInTheDocument();
    expect(screen.getByText("Of course!")).toBeInTheDocument();
  });

  it("should handle mixed system message content types", () => {
    const textContent = {
      type: "text" as const,
      text: "This is structured text content",
    };

    const messages = [
      {
        role: "system" as const,
        content: [{ type: "text" as const, text: "Plain string content" }],
      },
      { role: "system" as const, content: [textContent] },
      {
        role: "assistant" as const,
        content: [{ type: "text" as const, text: "Response" }],
        tool_calls: [],
      },
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("Plain string content")).toBeInTheDocument();
    expect(
      screen.getByText("This is structured text content"),
    ).toBeInTheDocument();
    expect(screen.getByText("Response")).toBeInTheDocument();
  });

  it("should handle system messages alongside character roleplay", () => {
    const assumeAlice: ToolCallFinished = {
      type: "finished",
      tool_name: "assume_character",
      tool_id: "call_1",
      parameters: {
        character_name: "Alice",
        personality: "cheerful",
      },
      result: {
        type: "success",
        content: { type: "text", text: "Character created" },
      },
    };

    const setMood: ToolCallFinished = {
      type: "finished",
      tool_name: "set_mood",
      tool_id: "call_2",
      parameters: { mood: "happy" },
      result: { type: "success", content: { type: "text", text: "Mood set" } },
    };

    const messages = [
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "I'm Alice" }],
        tool_calls: [assumeAlice],
      } as AgentMessage,
      {
        role: "system" as const,
        content: [
          { type: "text" as const, text: "Context saved automatically" },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text" as const, text: "Hello there!" }],
        tool_calls: [setMood],
      } as AgentMessage,
    ];

    render(
      <RoleplayPresenter
        messages={messages}
        isStreamActive={false}
        agentState={mockAgentState}
      />,
    );

    expect(screen.getByText("Context saved automatically")).toBeInTheDocument();
    expect(screen.getByText("Hello there!")).toBeInTheDocument();
    expect(screen.getByText("I'm Alice")).toBeInTheDocument();
  });
});
