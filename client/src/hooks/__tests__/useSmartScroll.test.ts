import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSmartScroll } from "../useSmartScroll";

// Mock scrollIntoView since it's not available in jsdom
const mockScrollIntoView = vi.fn();
Object.defineProperty(Element.prototype, "scrollIntoView", {
  value: mockScrollIntoView,
  writable: true,
});

describe("useSmartScroll", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should initialize with user at bottom", () => {
    const { result } = renderHook(() => useSmartScroll({ items: [] }));

    expect(result.current.isUserAtBottom).toBe(true);
  });

  it("should provide refs for messages container and end", () => {
    const { result } = renderHook(() => useSmartScroll({ items: [] }));

    expect(result.current.messagesContainerRef).toBeDefined();
    expect(result.current.messagesEndRef).toBeDefined();
  });

  it("should update isUserAtBottom based on scroll position", () => {
    const { result } = renderHook(() =>
      useSmartScroll({ items: [], threshold: 100 }),
    );

    // Mock scroll event - user is at bottom
    const scrollEvent = {
      currentTarget: {
        scrollTop: 400,
        clientHeight: 600,
        scrollHeight: 1000, // 400 + 600 = 1000, exactly at bottom
      },
    } as React.UIEvent<HTMLDivElement>;

    act(() => {
      result.current.handleScroll(scrollEvent);
    });

    expect(result.current.isUserAtBottom).toBe(true);
  });

  it("should detect when user is not at bottom", () => {
    const { result } = renderHook(() =>
      useSmartScroll({ items: [], threshold: 100 }),
    );

    // Mock scroll event - user is not at bottom (more than threshold away)
    const scrollEvent = {
      currentTarget: {
        scrollTop: 200,
        clientHeight: 600,
        scrollHeight: 1000, // 200 + 600 = 800, which is 200px from bottom (> 100px threshold)
      },
    } as React.UIEvent<HTMLDivElement>;

    act(() => {
      result.current.handleScroll(scrollEvent);
    });

    expect(result.current.isUserAtBottom).toBe(false);
  });

  it("should consider user at bottom when within threshold", () => {
    const { result } = renderHook(() =>
      useSmartScroll({ items: [], threshold: 100 }),
    );

    // Mock scroll event - user is within threshold of bottom
    const scrollEvent = {
      currentTarget: {
        scrollTop: 350,
        clientHeight: 600,
        scrollHeight: 1000, // 350 + 600 = 950, which is 50px from bottom (< 100px threshold)
      },
    } as React.UIEvent<HTMLDivElement>;

    act(() => {
      result.current.handleScroll(scrollEvent);
    });

    expect(result.current.isUserAtBottom).toBe(true);
  });

  it("should use custom threshold", () => {
    const { result } = renderHook(() =>
      useSmartScroll({ items: [], threshold: 200 }),
    );

    // Mock scroll event - 150px from bottom
    const scrollEvent = {
      currentTarget: {
        scrollTop: 250,
        clientHeight: 600,
        scrollHeight: 1000, // 250 + 600 = 850, which is 150px from bottom
      },
    } as React.UIEvent<HTMLDivElement>;

    act(() => {
      result.current.handleScroll(scrollEvent);
    });

    // Should be considered at bottom with 200px threshold
    expect(result.current.isUserAtBottom).toBe(true);
  });

  it("should scroll to bottom when items change and user is at bottom", () => {
    const { result, rerender } = renderHook(
      ({ items }) => useSmartScroll({ items }),
      { initialProps: { items: ["item1"] } },
    );

    // Mock the ref to have a current element
    const mockElement = document.createElement("div");
    mockElement.scrollIntoView = mockScrollIntoView;
    result.current.messagesEndRef.current = mockElement;

    // Ensure user is at bottom
    expect(result.current.isUserAtBottom).toBe(true);

    // Add new item
    rerender({ items: ["item1", "item2"] });

    expect(mockScrollIntoView).toHaveBeenCalledWith({ behavior: "smooth" });
  });

  it("should not scroll when items change but user is not at bottom", () => {
    const { result, rerender } = renderHook(
      ({ items }) => useSmartScroll({ items }),
      { initialProps: { items: ["item1"] } },
    );

    // Set user not at bottom
    act(() => {
      result.current.setUserAtBottom(false);
    });

    // Add new item
    rerender({ items: ["item1", "item2"] });

    expect(mockScrollIntoView).not.toHaveBeenCalled();
  });

  it("should provide scrollToBottom function", () => {
    const { result } = renderHook(() => useSmartScroll({ items: [] }));

    // Mock the ref to have a current element
    const mockElement = document.createElement("div");
    mockElement.scrollIntoView = mockScrollIntoView;
    result.current.messagesEndRef.current = mockElement;

    act(() => {
      result.current.scrollToBottom();
    });

    expect(mockScrollIntoView).toHaveBeenCalledWith({ behavior: "smooth" });
  });

  it("should provide setUserAtBottom function", () => {
    const { result } = renderHook(() => useSmartScroll({ items: [] }));

    act(() => {
      result.current.setUserAtBottom(false);
    });

    expect(result.current.isUserAtBottom).toBe(false);

    act(() => {
      result.current.setUserAtBottom(true);
    });

    expect(result.current.isUserAtBottom).toBe(true);
  });
});
