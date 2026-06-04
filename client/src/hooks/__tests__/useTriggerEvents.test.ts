import { renderHook } from "@testing-library/react";
import { useTriggerEvents } from "../useTriggerEvents";
import { ClientAgentEvent } from "../useWebSocket";

const imageResult = {
  type: "image_generated" as const,
  prompt: "An ethereal being with flowing robes",
  chunks: null,
  image_path: "generated.png",
  image_url: "http://example.com/image.png",
  width: 512,
  height: 512,
  num_inference_steps: 20,
  guidance_scale: 7.5,
  negative_prompt: null,
  original_description: null,
  optimization_confidence: null,
  camera_angle: null,
  viewpoint: null,
  optimization_notes: null,
};

describe("useTriggerEvents", () => {
  it("should start with empty streaming entries", () => {
    const { result } = renderHook(() => useTriggerEvents([]));

    expect(result.current.streamingEntries).toEqual([]);
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
          image_paths: null,
        },
        entry_id: "entry_1",
        timestamp: "2024-01-01T10:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_1",
        action_type: "speak",
        sequence_number: 1,
        action_number: 1,
        reasoning: "The user is asking about my well-being.",
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
          reasoning: "The user is asking about my well-being.",
          input: {
            intent: "Respond warmly to the user's greeting",
            tone: null,
          },
          result: {
            type: "success",
            content: {
              response: "Hello! I'm doing well, thanks for asking!",
            },
          },
          duration_ms: 1500,
          start_timestamp: "2024-01-01T10:00:01Z",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T10:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_1",
          trigger: {
            type: "user_input",
            content: "Hello, how are you?",
            user_name: "TestUser",
            timestamp: "2024-01-01T10:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "speak",
              reasoning: "The user is asking about my well-being.",
              input: {
                intent: "Respond warmly to the user's greeting",
                tone: null,
              },
              result: {
                type: "success",
                content: {
                  response: "Hello! I'm doing well, thanks for asking!",
                },
              },
              duration_ms: 1500,
              start_timestamp: "2024-01-01T10:00:01Z",
            },
          ],
          timestamp: "2024-01-01T10:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 50,
        context_limit: 500,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    expect(result.current.isStreamActive).toBe(false);

    const trigger = result.current.streamingEntries[0];
    expect(trigger.entry_id).toBe("entry_1");
    expect(trigger.trigger.type).toBe("user_input");
    if (trigger.trigger.type === "user_input") {
      expect(trigger.trigger.content).toBe("Hello, how are you?");
    }
    expect(trigger.actions_taken).toHaveLength(1);

    const action = trigger.actions_taken[0];
    expect(action.type).toBe("speak");
    expect(action.result.type).toBe("success");
    if (action.type === "speak" && action.result.type === "success") {
      expect(action.result.content.response).toBe(
        "Hello! I'm doing well, thanks for asking!",
      );
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
          image_paths: null,
        },
        entry_id: "entry_2",
        timestamp: "2024-01-01T11:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_2",
        action_type: "think",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T11:00:01Z",
        reasoning: "I want to provide a friendly and authentic response.",
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

    expect(result.current.streamingEntries).toHaveLength(1);
    expect(result.current.isStreamActive).toBe(true);

    const activeTrigger = result.current.streamingEntries[0];
    expect(activeTrigger.entry_id).toBe("entry_2");
    if (activeTrigger.trigger.type === "user_input") {
      expect(activeTrigger.trigger.content).toBe("Tell me about yourself");
    }
    expect(activeTrigger.actions_taken).toHaveLength(1);

    const activeAction = activeTrigger.actions_taken[0];
    expect(activeAction.type).toBe("think");
    expect(activeAction.result.type).toBe("streaming");
    if (activeAction.result.type === "streaming") {
      expect(activeAction.result.result).toBe(
        "I should be authentic and friendly...",
      );
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
          image_paths: null,
        },
        entry_id: "entry_3",
        timestamp: "2024-01-01T12:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "think",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T12:00:01Z",
        reasoning: "I should update my mood and let the user know.",
      },
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "update_mood",
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T12:00:02Z",
        reasoning: "I should update my mood to happy.",
      },
      {
        id: 3,
        type: "action_started",
        entry_id: "entry_3",
        action_type: "speak",
        sequence_number: 1,
        action_number: 3,
        timestamp: "2024-01-01T12:00:03Z",
        reasoning: "I want to acknowledge the user's mood change.",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_3",
        action: {
          type: "think",
          reasoning: "I should update my mood and let the user know.",
          input: { focus: "Consider the mood change" },
          result: {
            type: "success",
            content: { thoughts: "I should update my mood and let the user know" },
          },
          duration_ms: 500,
          start_timestamp: "2024-01-01T12:00:01Z",
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
          reasoning: "I should update my mood to happy.",
          input: {
            reason: "Update mood to happy",
            new_mood: "happy",
            intensity: "medium",
          },
          result: {
            type: "success",
            content: {
              old_mood: "neutral",
              old_intensity: "medium",
              new_mood: "happy",
              new_intensity: "medium",
              reason: "Mood updated to happy",
            },
          },
          duration_ms: 200,
          start_timestamp: "2024-01-01T12:00:02Z",
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
          reasoning: "I want to acknowledge the user's mood change.",
          input: { intent: "Acknowledge the mood change", tone: null },
          result: {
            type: "success",
            content: { response: "Great! I'm feeling happy now!" },
          },
          duration_ms: 800,
          start_timestamp: "2024-01-01T12:00:03Z",
        },
        sequence_number: 1,
        action_number: 3,
        timestamp: "2024-01-01T12:00:06Z",
      },
      {
        id: 7,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_3",
          trigger: {
            type: "user_input",
            content: "Change your mood to happy",
            user_name: "TestUser",
            timestamp: "2024-01-01T12:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "think",
              reasoning: "I should update my mood and let the user know.",
              input: { focus: "Consider the mood change" },
              result: {
                type: "success",
                content: {
                  thoughts: "I should update my mood and let the user know",
                },
              },
              duration_ms: 500,
              start_timestamp: "2024-01-01T12:00:01Z",
            },
            {
              type: "update_mood",
              reasoning: "I should update my mood to happy.",
              input: {
                reason: "Update mood to happy",
                new_mood: "happy",
                intensity: "medium",
              },
              result: {
                type: "success",
                content: {
                  old_mood: "neutral",
                  old_intensity: "medium",
                  new_mood: "happy",
                  new_intensity: "medium",
                  reason: "Mood updated to happy",
                },
              },
              duration_ms: 200,
              start_timestamp: "2024-01-01T12:00:02Z",
            },
            {
              type: "speak",
              reasoning: "I want to acknowledge the user's mood change.",
              input: { intent: "Acknowledge the mood change", tone: null },
              result: {
                type: "success",
                content: { response: "Great! I'm feeling happy now!" },
              },
              duration_ms: 800,
              start_timestamp: "2024-01-01T12:00:03Z",
            },
          ],
          timestamp: "2024-01-01T12:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    const trigger = result.current.streamingEntries[0];

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
          image_paths: null,
        },
        entry_id: "entry_4",
        timestamp: "2024-01-01T13:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_4",
        action_type: "think",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T13:00:01Z",
        reasoning: "First action reasoning",
      },
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_4",
        action_type: "speak",
        sequence_number: 1,
        action_number: 2,
        timestamp: "2024-01-01T13:00:02Z",
        reasoning: "Second action reasoning",
      },
      // Completion events arrive out of order
      {
        id: 3,
        type: "action_completed",
        entry_id: "entry_4",
        action: {
          type: "speak",
          reasoning: "Second action reasoning",
          input: { intent: "Second action", tone: null },
          result: {
            type: "success",
            content: { response: "Second action result" },
          },
          duration_ms: 300,
          start_timestamp: "2024-01-01T13:00:02Z",
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
          reasoning: "First action reasoning",
          input: { focus: "First action" },
          result: {
            type: "success",
            content: { thoughts: "First action result" },
          },
          duration_ms: 800,
          start_timestamp: "2024-01-01T13:00:01Z",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T13:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_4",
          trigger: {
            type: "user_input",
            content: "Do something complex",
            user_name: "TestUser",
            timestamp: "2024-01-01T13:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "think",
              reasoning: "First action reasoning",
              input: { focus: "First action" },
              result: {
                type: "success",
                content: { thoughts: "First action result" },
              },
              duration_ms: 800,
              start_timestamp: "2024-01-01T13:00:01Z",
            },
            {
              type: "speak",
              reasoning: "Second action reasoning",
              input: { intent: "Second action", tone: null },
              result: {
                type: "success",
                content: { response: "Second action result" },
              },
              duration_ms: 300,
              start_timestamp: "2024-01-01T13:00:02Z",
            },
          ],
          timestamp: "2024-01-01T13:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 50,
        context_limit: 500,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    const trigger = result.current.streamingEntries[0];

    expect(trigger.actions_taken).toHaveLength(2);

    // Actions should be sorted by (sequence_number, action_number) not completion order
    expect(trigger.actions_taken[0].type).toBe("think"); // sequence 1, action 1
    expect(trigger.actions_taken[1].type).toBe("speak"); // sequence 1, action 2

    const firstAction = trigger.actions_taken[0];
    const secondAction = trigger.actions_taken[1];
    if (firstAction.type === "think" && firstAction.result.type === "success") {
      expect(firstAction.result.content.thoughts).toBe("First action result");
    }
    if (secondAction.type === "speak" && secondAction.result.type === "success") {
      expect(secondAction.result.content.response).toBe("Second action result");
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
          image_paths: null,
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
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T14:00:01Z",
        reasoning: "First sequence reasoning",
      },
      {
        id: 2,
        type: "action_completed",
        entry_id: "entry_5",
        action: {
          type: "think",
          reasoning: "First sequence reasoning",
          input: { focus: "First sequence thinking" },
          result: {
            type: "success",
            content: { thoughts: "First sequence thought" },
          },
          duration_ms: 400,
          start_timestamp: "2024-01-01T14:00:01Z",
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
        sequence_number: 2,
        action_number: 1,
        timestamp: "2024-01-01T14:00:03Z",
        reasoning: "Second action reasoning",
      },
      {
        id: 4,
        type: "action_completed",
        entry_id: "entry_5",
        action: {
          type: "speak",
          reasoning: "Second action reasoning",
          input: { intent: "Second sequence speaking", tone: null },
          result: {
            type: "success",
            content: { response: "Second sequence response" },
          },
          duration_ms: 600,
          start_timestamp: "2024-01-01T14:00:03Z",
        },
        sequence_number: 2,
        action_number: 1,
        timestamp: "2024-01-01T14:00:04Z",
      },
      {
        id: 5,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_5",
          trigger: {
            type: "user_input",
            content: "Multi-sequence task",
            user_name: "TestUser",
            timestamp: "2024-01-01T14:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "think",
              reasoning: "First sequence reasoning",
              input: { focus: "First sequence thinking" },
              result: {
                type: "success",
                content: { thoughts: "First sequence thought" },
              },
              duration_ms: 400,
              start_timestamp: "2024-01-01T14:00:01Z",
            },
            {
              type: "speak",
              reasoning: "Second action reasoning",
              input: { intent: "Second sequence speaking", tone: null },
              result: {
                type: "success",
                content: { response: "Second sequence response" },
              },
              duration_ms: 600,
              start_timestamp: "2024-01-01T14:00:03Z",
            },
          ],
          timestamp: "2024-01-01T14:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 75,
        context_limit: 750,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    const trigger = result.current.streamingEntries[0];

    expect(trigger.actions_taken).toHaveLength(2);

    // Actions should be sorted: sequence 1 before sequence 2
    expect(trigger.actions_taken[0].type).toBe("think"); // sequence 1, action 1
    expect(trigger.actions_taken[1].type).toBe("speak"); // sequence 2, action 1
  });

  it("should ignore unknown event types", () => {
    const unknownEvent: ClientAgentEvent = {
      // @ts-expect-error Deliberately invalid event to preserve this runtime behavior test.
      type: "unknown_event",
      id: 1,
    };
    const events: ClientAgentEvent[] = [
      {
        id: 0,
        type: "trigger_started",
        trigger: {
          type: "user_input",
          content: "Test",
          user_name: "TestUser",
          timestamp: "2024-01-01T15:00:00Z",
          image_paths: null,
        },
        entry_id: "entry_6",
        timestamp: "2024-01-01T15:00:00Z",
      },
      // Unknown event type
      unknownEvent,
      {
        id: 2,
        type: "action_started",
        entry_id: "entry_6",
        action_type: "speak",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T15:00:01Z",
        reasoning: "",
      },
      {
        id: 3,
        type: "action_completed",
        entry_id: "entry_6",
        action: {
          type: "speak",
          reasoning: "",
          input: { intent: "Test action", tone: null },
          result: {
            type: "success",
            content: { response: "Test result" },
          },
          duration_ms: 100,
          start_timestamp: "2024-01-01T15:00:01Z",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T15:00:02Z",
      },
      {
        id: 4,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_6",
          trigger: {
            type: "user_input",
            content: "Test",
            user_name: "TestUser",
            timestamp: "2024-01-01T15:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "speak",
              reasoning: "",
              input: { intent: "Test action", tone: null },
              result: {
                type: "success",
                content: { response: "Test result" },
              },
              duration_ms: 100,
              start_timestamp: "2024-01-01T15:00:01Z",
            },
          ],
          timestamp: "2024-01-01T15:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    expect(result.current.streamingEntries[0].actions_taken).toHaveLength(1);
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
          image_paths: null,
        },
        entry_id: "entry_7",
        timestamp: "2024-01-01T16:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_7",
        action_type: "speak",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T16:00:01Z",
        reasoning: "Valid reasoning",
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
          reasoning: "Valid reasoning",
          input: { intent: "Valid context", tone: null },
          result: {
            type: "success",
            content: { response: "Valid result" },
          },
          duration_ms: 200,
          start_timestamp: "2024-01-01T16:00:01Z",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T16:00:03Z",
      },
      {
        id: 4,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_7",
          trigger: {
            type: "user_input",
            content: "Valid trigger",
            user_name: "TestUser",
            timestamp: "2024-01-01T16:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "speak",
              reasoning: "Valid reasoning",
              input: { intent: "Valid context", tone: null },
              result: {
                type: "success",
                content: { response: "Valid result" },
              },
              duration_ms: 200,
              start_timestamp: "2024-01-01T16:00:01Z",
            },
          ],
          timestamp: "2024-01-01T16:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    const trigger = result.current.streamingEntries[0];

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
          image_paths: null,
        },
        entry_id: "entry_8",
        timestamp: "2024-01-01T17:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_8",
        action_type: "speak",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T17:00:01Z",
        reasoning: "Test reasoning",
      },
    ];

    const { result, rerender } = renderHook(
      ({ events }) => useTriggerEvents(events),
      {
        initialProps: { events },
      },
    );

    expect(result.current.streamingEntries).toHaveLength(1);
    expect(result.current.streamingEntries[0].actions_taken).toHaveLength(1);

    // Add the same events again (simulating duplicate WebSocket messages)
    events = [
      ...events,
      ...events, // Duplicate all events
    ];

    rerender({ events });

    // Should still only have one trigger with one action
    expect(result.current.streamingEntries).toHaveLength(1);
    expect(result.current.streamingEntries[0].actions_taken).toHaveLength(1);
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
          image_paths: null,
        },
        entry_id: "entry_9",
        timestamp: "2024-01-01T18:00:00Z",
      },
      {
        id: 1,
        type: "action_started",
        entry_id: "entry_9",
        action_type: "update_appearance",
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T18:00:01Z",
        reasoning: "Need to refresh appearance to match new theme.",
      },
      {
        id: 2,
        type: "action_completed",
        entry_id: "entry_9",
        action: {
          type: "update_appearance",
          reasoning: "Need to refresh appearance to match new theme.",
          input: {
            reason: "Update appearance with new image",
            change_description: "An ethereal being with flowing robes",
          },
          result: {
            type: "success",
            content: {
              image_description: "An ethereal being with flowing robes",
              old_appearance: "Previous appearance",
              new_appearance: "Appearance updated with new ethereal look",
              reason: "Update appearance with new image",
              image_result: imageResult,
            },
          },
          duration_ms: 2000,
          start_timestamp: "2024-01-01T18:00:01Z",
        },
        sequence_number: 1,
        action_number: 1,
        timestamp: "2024-01-01T18:00:03Z",
      },
      {
        id: 3,
        type: "trigger_completed",
        entry: {
          entry_id: "entry_9",
          trigger: {
            type: "user_input",
            content: "Update your appearance",
            user_name: "TestUser",
            timestamp: "2024-01-01T18:00:00Z",
            image_paths: null,
          },
          actions_taken: [
            {
              type: "update_appearance",
              reasoning: "Need to refresh appearance to match new theme.",
              input: {
                reason: "Update appearance with new image",
                change_description: "An ethereal being with flowing robes",
              },
              result: {
                type: "success",
                content: {
                  image_description: "An ethereal being with flowing robes",
                  old_appearance: "Previous appearance",
                  new_appearance: "Appearance updated with new ethereal look",
                  reason: "Update appearance with new image",
                  image_result: imageResult,
                },
              },
              duration_ms: 2000,
              start_timestamp: "2024-01-01T18:00:01Z",
            },
          ],
          timestamp: "2024-01-01T18:00:00Z",
          end_timestamp: null,
          situational_context: "Test situational context",
          compressed_summary: null,        },
        estimated_tokens: 100,
        context_limit: 1000,
        usage_percentage: 10,
        approaching_limit: false,
      },
    ];

    const { result } = renderHook(() => useTriggerEvents(events));

    expect(result.current.streamingEntries).toHaveLength(1);
    const trigger = result.current.streamingEntries[0];

    expect(trigger.actions_taken).toHaveLength(1);
    const action = trigger.actions_taken[0];

    expect(action.type).toBe("update_appearance");
    if (action.type === "update_appearance" && action.result.type === "success") {
      expect(action.result.content.image_description).toBe(
        "An ethereal being with flowing robes",
      );
      expect(action.result.content.image_result.image_url).toBe(
        "http://example.com/image.png",
      );
    }
  });
});
