import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ChatMessages } from "../ChatMessages";

describe("ChatMessages", () => {
  const mockUserMessage = {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Hello world" }],
  };

  const mockAgentMessage = {
    role: "assistant" as const,
    content: [{ type: "text" as const, text: "Hi there!" }],
    tool_calls: [],
  };

  it("should render empty state when no items", () => {
    render(<ChatMessages messages={[]} />);

    expect(
      screen.getByText("Start a conversation with the agent!"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Try: "Please roleplay as Elena, a mysterious vampire."',
      ),
    ).toBeInTheDocument();
  });

  it("should render text items", () => {
    render(<ChatMessages messages={[mockUserMessage]} />);

    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("should render tool items", () => {
    render(<ChatMessages messages={[mockAgentMessage]} />);

    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should render multiple mixed items", () => {
    const messages = [mockUserMessage, mockAgentMessage];
    render(<ChatMessages messages={messages} />);

    expect(screen.getByText("Hello world")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should show streaming cursor when not complete and has items", () => {
    render(<ChatMessages messages={[mockUserMessage]} isStreamActive={true} />);

    const cursor = screen.getByText("▋");
    expect(cursor).toBeInTheDocument();
    expect(cursor).toHaveClass("animate-pulse", "text-gray-500");
  });

  it("should not show streaming cursor when complete", () => {
    render(
      <ChatMessages messages={[mockUserMessage]} isStreamActive={false} />,
    );

    expect(screen.queryByText("▋")).not.toBeInTheDocument();
  });

  it("should not show streaming cursor when no items", () => {
    render(<ChatMessages messages={[]} isStreamActive={true} />);

    expect(screen.queryByText("▋")).not.toBeInTheDocument();
  });

  it("should call onScroll when scrolled", () => {
    const onScroll = vi.fn();
    const { container } = render(
      <ChatMessages messages={[mockUserMessage]} onScroll={onScroll} />,
    );

    const scrollContainer = container.firstChild as HTMLElement;
    fireEvent.scroll(scrollContainer);

    expect(onScroll).toHaveBeenCalledTimes(1);
  });

  it("should apply custom className", () => {
    const { container } = render(
      <ChatMessages messages={[]} className="custom-class" />,
    );

    const scrollContainer = container.firstChild as HTMLElement;
    expect(scrollContainer).toHaveClass("custom-class");
  });

  it("should apply default classes", () => {
    const { container } = render(<ChatMessages messages={[]} />);

    const scrollContainer = container.firstChild as HTMLElement;
    expect(scrollContainer).toHaveClass(
      "flex-1",
      "overflow-y-auto",
      "px-4",
      "py-4",
      "space-y-4",
    );
  });

  it("should forward ref correctly", () => {
    const ref = vi.fn();
    render(<ChatMessages ref={ref} messages={[]} />);

    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });

  it("should render items with correct prose styling", () => {
    const { container } = render(<ChatMessages messages={[mockUserMessage]} />);

    // Look for the prose wrapper div
    const proseElements = container.querySelectorAll(
      ".prose.prose-sm.max-w-none",
    );
    expect(proseElements).toHaveLength(1);
  });

  it("should render streaming cursor with correct prose styling", () => {
    render(<ChatMessages messages={[mockUserMessage]} isStreamActive={true} />);

    const cursorContainer = screen.getByText("▋").closest("div");
    expect(cursorContainer).toHaveClass("prose", "prose-sm", "max-w-none");
  });
});
