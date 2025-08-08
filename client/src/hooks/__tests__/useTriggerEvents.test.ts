import { renderHook } from "@testing-library/react";
import { useTriggerEvents } from "../useTriggerEvents";
import { ClientAgentEvent } from "../useWebSocket";

describe("useTriggerEvents", () => {
  it("should start with empty trigger entries", () => {
    const { result } = renderHook(() => useTriggerEvents([]));

    expect(result.current.triggerEntries).toEqual([]);
    expect(result.current.isStreamActive).toBe(false);
  });

  it("should handle a complete trigger flow with single action", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Hello, how are you?",
          user_name: "TestUser",
          timestamp: "2024-01-01T10:00:00Z",
        },
        entry_id: "entry_1",
        timestamp: "2024-01-01T10:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_1",
        action_type: "speak",
        context_given: "Respond warmly to the user's greeting",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T10:00:01Z",
      },
      {
        id: 2,
        type: "action_progress",
        entry_id: "entry_1",
        action_type: "speak",
        partial_result: "Hello! ",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T10:00:02Z",
      },
      {
        id: 3,
        type: "action_progress",
        entry_id: "entry_1",
        action_type: "speak",
        partial_result: "I'm doing well, thanks for asking!",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T10:00:03Z",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_1",
        action: {
          type: "speak",
          context_given: "Respond warmly to the user's greeting",
          status: {
            type: "success",
            result: "Hello! I'm doing well, thanks for asking!",
          },
          duration_ms: 1500,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T10:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry_id: "entry_1",
        total_actions: 1,
        successful_actions: 1,
        timestamp: "2024-01-01T10:00:05Z",
        estimated_tokens: 50,
        context_limit: 500,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    expect(result.current.isStreamActive).toBe(false);

    const trigger = result.current.triggerEntries[0];
    expect(trigger.entry_id).toBe("entry_1");
    expect(trigger.trigger.type).toBe("user_input");
    expect(trigger.trigger.content).toBe("Hello, how are you?");
    expect(trigger.actions_taken).toHaveLength(1);

    const action = trigger.actions_taken[0];
    console.log(action);
    expect(action.type).toBe("speak");
    expect(action.status.type).toBe("success");
    if (action.status.type === "success") {
      expect(action.status.result).toBe("Hello! I'm doing well, thanks for asking!");
    }
    expect(action.duration_ms).toBe(1500);
  });

  it("should maintain active trigger during streaming", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Tell me about yourself",
          user_name: "TestUser",
          timestamp: "2024-01-01T11:00:00Z",
        },
        entry_id: "entry_2",
        timestamp: "2024-01-01T11:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_2",
        action_type: "think",
        context_given: "Consider what to share about myself",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T11:00:01Z",
      },
      {
        id: 2,
        type: "action_progress",
        entry_id: "entry_2",
        action_type: "think",
        partial_result: "I should be authentic and friendly...",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T11:00:02Z",
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    expect(result.current.isStreamActive).toBe(true);

    const activeTrigger = result.current.triggerEntries[0];
    expect(activeTrigger.entry_id).toBe("entry_2");
    expect(activeTrigger.trigger.content).toBe("Tell me about yourself");
    expect(activeTrigger.actions_taken).toHaveLength(1);

    const activeAction = activeTrigger.actions_taken[0];
    expect(activeAction.type).toBe("think");
    expect(activeAction.status.type).toBe("streaming");
    if (activeAction.status.type === "streaming") {
      expect(activeAction.status.result).toBe("I should be authentic and friendly...");
    }
  });

  it("should handle multiple actions in correct order", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Change your mood to happy",
          user_name: "TestUser",
          timestamp: "2024-01-01T12:00:00Z",
        },
        entry_id: "entry_3",
        timestamp: "2024-01-01T12:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "think",
        context_given: "Consider the mood change",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T12:00:01Z",
      },
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "update_mood",
        context_given: "Update mood to happy",
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T12:00:02Z",
      },
      {
        id: 3,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "speak",
        context_given: "Acknowledge the mood change",
        sequence_number: 1,
        action_number: 3,
        timestamp: "2024-01-01T12:00:03Z",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_3",
        action: {
          type: "think",
          context_given: "Consider the mood change",
          status: {
            type: "success",
            result: "I should update my mood and let the user know",
          },
          duration_ms: 500,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T12:00:04Z",
      },
      {
        id: 5,
        type: "action_completed",
        entry_id: "entry_3",
        action: {
          type: "update_mood",
          context_given: "Update mood to happy",
          status: {
            type: "success",
            result: "Mood updated to happy",
          },
          duration_ms: 200,
        },
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T12:00:05Z",
      },
      {
        id: 6,
        type: "action_completed",
        entry_id: "entry_3",
        action: {
          type: "speak",
          context_given: "Acknowledge the mood change",
          status: {
            type: "success",
            result: "Great! I'm feeling happy now! 😊",
          },
          duration_ms: 800,
        },
        sequence_number: 1,
        action_number: 3,
        timestamp: "2024-01-01T12:00:06Z",
      },
      {
        id: 7,
        type: "trigger_completed",
        entry_id: "entry_3",
        total_actions: 3,
        successful_actions: 3,
        timestamp: "2024-01-01T12:00:07Z",
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    const trigger = result.current.triggerEntries[0];

    expect(trigger.actions_taken).toHaveLength(3);

    // Actions should be in execution order
    expect(trigger.actions_taken[0].type).toBe("think");
    expect(trigger.actions_taken[1].type).toBe("update_mood");
    expect(trigger.actions_taken[2].type).toBe("speak");
  });

  it("should handle out-of-order events correctly", () => {
    // Simulate events arriving out of order
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Do something complex",
          user_name: "TestUser",
          timestamp: "2024-01-01T13:00:00Z",
        },
        entry_id: "entry_4",
        timestamp: "2024-01-01T13:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_4",
        action_type: "think",
        context_given: "First action",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T13:00:01Z",
      },
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_4",
        action_type: "speak",
        context_given: "Second action",
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T13:00:02Z",
      },
      // Completion events arrive out of order
      {
        id: 3,
        type: "action_completed",
        entry_id: "entry_4",
        action: {
          type: "speak",
          context_given: "Second action",
          status: {
            type: "success",
            result: "Second action result",
          },
          duration_ms: 300,
        },
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T13:00:05Z",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_4",
        action: {
          type: "think",
          context_given: "First action",
          status: {
            type: "success",
            result: "First action result",
          },
          duration_ms: 800,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T13:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry_id: "entry_4",
        total_actions: 2,
        successful_actions: 2,
        timestamp: "2024-01-01T13:00:06Z",
        estimated_tokens: 50,
        context_limit: 500,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    const trigger = result.current.triggerEntries[0];

    expect(trigger.actions_taken).toHaveLength(2);

    // Actions should be sorted by (sequence_number, action_number) not completion order
    expect(trigger.actions_taken[0].type).toBe("think"); // sequence 1, action 1
    expect(trigger.actions_taken[1].type).toBe("speak"); // sequence 1, action 2

    const firstAction = trigger.actions_taken[0];
    const secondAction = trigger.actions_taken[1];
    if (firstAction.status.type === "success") {
      expect(firstAction.status.result).toBe("First action result");
    }
    if (secondAction.status.type === "success") {
      expect(secondAction.status.result).toBe("Second action result");
    }
  });

  it("should handle multiple sequences correctly", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Multi-sequence task",
          user_name: "TestUser",
          timestamp: "2024-01-01T14:00:00Z",
        },
        entry_id: "entry_5",
        timestamp: "2024-01-01T14:00:00Z",
      },
      // First sequence
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_5",
        action_type: "think",
        context_given: "First sequence thinking",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T14:00:01Z",
      },
      {
        id: 2,
        type: "action_completed",
        entry_id: "entry_5",
        action: {
          type: "think",
          context_given: "First sequence thinking",
          status: {
            type: "success",
            result: "First sequence thought",
          },
          duration_ms: 400,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T14:00:02Z",
      },
      // Second sequence
      {
        id: 3,
        type: "action_started",
        entry_id: "entry_5",
        action_type: "speak",
        context_given: "Second sequence speaking",
        sequence_number: 2,
        action_number: 1,
        timestamp: "2024-01-01T14:00:03Z",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_5",
        action: {
          type: "speak",
          context_given: "Second sequence speaking",
          status: {
            type: "success",
            result: "Second sequence response",
          },
          duration_ms: 600,
        },
        sequence_number: 2,
        action_number: 1,
        timestamp: "2024-01-01T14:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry_id: "entry_5",
        total_actions: 2,
        successful_actions: 2,
        timestamp: "2024-01-01T14:00:05Z",
        estimated_tokens: 75,
        context_limit: 750,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    const trigger = result.current.triggerEntries[0];

    expect(trigger.actions_taken).toHaveLength(2);

    // Actions should be sorted: sequence 1 before sequence 2
    expect(trigger.actions_taken[0].type).toBe("think"); // sequence 1, action 1
    expect(trigger.actions_taken[1].type).toBe("speak"); // sequence 2, action 1
  });

  it("should ignore unknown event types", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Test",
          user_name: "TestUser",
          timestamp: "2024-01-01T15:00:00Z",
        },
        entry_id: "entry_6",
        timestamp: "2024-01-01T15:00:00Z",
      },
      // Unknown event type
      {
        id: 1,
        type: "unknown_event" as any,
      } as any,
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_6",
        action_type: "speak",
        context_given: "Test action",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T15:00:01Z",
      },
      {
        id: 3,
        type: "action_completed",
        entry_id: "entry_6",
        action: {
          type: "speak",
          context_given: "Test action",
          status: {
            type: "success",
            result: "Test result",
          },
          duration_ms: 100,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T15:00:02Z",
      },
      {
        id: 4,
        type: "trigger_completed",
        entry_id: "entry_6",
        total_actions: 1,
        successful_actions: 1,
        timestamp: "2024-01-01T15:00:03Z",
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    expect(result.current.triggerEntries[0].actions_taken).toHaveLength(1);
  });

  it("should handle events with mismatched entry_ids", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Valid trigger",
          user_name: "TestUser",
          timestamp: "2024-01-01T16:00:00Z",
        },
        entry_id: "entry_7",
        timestamp: "2024-01-01T16:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_7",
        action_type: "speak",
        context_given: "Valid action",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T16:00:01Z",
      },
      // Event with wrong entry_id - should be ignored
      {
        id: 2,
        type: "action_progress",
        entry_id: "wrong_entry_id",
        action_type: "speak",
        partial_result: "Should be ignored",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T16:00:02Z",
      },
      {
        id: 3,
        type: "action_completed",
        entry_id: "entry_7",
        action: {
          type: "speak",
          context_given: "Valid context",
          status: {
            type: "success",
            result: "Valid result",
          },
          duration_ms: 200,
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T16:00:03Z",
      },
      {
        id: 4,
        type: "trigger_completed",
        entry_id: "entry_7",
        total_actions: 1,
        successful_actions: 1,
        timestamp: "2024-01-01T16:00:04Z",
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    const trigger = result.current.triggerEntries[0];

    expect(trigger.actions_taken).toHaveLength(1);
    // The progress event with wrong entry_id should be ignored
    // So partial_results should be empty
    // (We can't directly check this since it's converted to final action)
  });

  it("should not process duplicate events", () => {
    let events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Test duplicate handling",
          user_name: "TestUser",
          timestamp: "2024-01-01T17:00:00Z",
        },
        entry_id: "entry_8",
        timestamp: "2024-01-01T17:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_8",
        action_type: "speak",
        context_given: "Test action",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T17:00:01Z",
      },
    ];

    const { result, rerender } = renderHook(
      ({ events }) => useTriggerEvents(events),
      {
        initialProps: { events },
      }
    );

    expect(result.current.triggerEntries).toHaveLength(1);
    expect(result.current.triggerEntries[0].actions_taken).toHaveLength(1);

    // Add the same events again (simulating duplicate WebSocket messages)
    events = [
      ...events,
      ...events, // Duplicate all events
    ];

    rerender({ events });

    // Should still only have one trigger with one action
    expect(result.current.triggerEntries).toHaveLength(1);
    expect(result.current.triggerEntries[0].actions_taken).toHaveLength(1);
  });

  it("should handle action with metadata correctly", () => {
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Update your appearance",
          user_name: "TestUser",
          timestamp: "2024-01-01T18:00:00Z",
        },
        entry_id: "entry_9",
        timestamp: "2024-01-01T18:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_9",
        action_type: "update_appearance",
        context_given: "Update appearance with new image",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T18:00:01Z",
      },
      {
        id: 2,
        type: "action_completed",
        entry_id: "entry_9",
        action: {
          type: "update_appearance",
          context_given: "Update appearance with new image",
          status: {
            type: "success",
            result: "Appearance updated with new ethereal look",
          },
          duration_ms: 2000,
          image_description: "An ethereal being with flowing robes",
          image_url: "http://example.com/image.png",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T18:00:03Z",
      },
      {
        id: 3,
        type: "trigger_completed",
        entry_id: "entry_9",
        total_actions: 1,
        successful_actions: 1,
        timestamp: "2024-01-01T18:00:04Z",
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.triggerEntries).toHaveLength(1);
    const trigger = result.current.triggerEntries[0];

    expect(trigger.actions_taken).toHaveLength(1);
    const action = trigger.actions_taken[0] as any;

    expect(action.type).toBe("update_appearance");
    expect(action.image_description).toBe("An ethereal being with flowing robes");
    expect(action.image_url).toBe("http://example.com/image.png");
  });
});