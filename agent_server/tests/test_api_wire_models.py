import json
from datetime import datetime
from pathlib import Path

from agent.api_types.events import ActionCompletedEvent, ActionStartedEvent
from agent.chain_of_action.action.actions.speak_action import (
    SpeakActionData,
    SpeakInput,
    SpeakOutput,
)
from agent.chain_of_action.action.base_action_data import ActionSuccessResult
from agent.chain_of_action.trigger import ImageFilePath, UserInputTrigger
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage.serializers import TriggerSerializer


def test_action_started_event_does_not_expose_context_given():
    event = ActionStartedEvent(
        entry_id="entry_1",
        action_type="speak",
        reasoning="Respond directly from the planned reasoning.",
        sequence_number=1,
        action_number=1,
        timestamp="2024-01-01T10:00:00Z",
    )

    payload = event.model_dump(mode="json")

    assert payload["reasoning"] == "Respond directly from the planned reasoning."
    assert "context_given" not in payload


def test_action_completed_event_carries_backend_action_data_shape():
    action = SpeakActionData(
        reasoning="Say hello.",
        input=SpeakInput(intent="Greet the user", tone=None),
        result=ActionSuccessResult(content=SpeakOutput(response="Hello!")),
        duration_ms=42,
        start_timestamp=datetime(2024, 1, 1, 10, 0, 0),
    )

    event = ActionCompletedEvent(
        entry_id="entry_1",
        action=action,
        sequence_number=1,
        action_number=1,
        timestamp="2024-01-01T10:00:01Z",
    )

    payload = event.model_dump(mode="json")

    assert payload["action"]["type"] == "speak"
    assert payload["action"]["input"] == {
        "intent": "Greet the user",
        "tone": None,
    }
    assert payload["action"]["result"] == {
        "type": "success",
        "content": {"response": "Hello!"},
    }
    assert "status" not in payload["action"]
    assert "context_given" not in payload["action"]


def test_generated_client_types_are_current_and_do_not_reintroduce_dto_fields():
    from scripts.generate_client_types import OUTPUT, main

    output_path = Path(OUTPUT)
    before = output_path.read_text(encoding="utf-8")

    main()
    after = output_path.read_text(encoding="utf-8")

    assert after == before
    assert "context_given" not in after
    assert "status:" not in after
    # exclude=True fields must not leak into generated client types.
    assert "embedding_vector" not in after
    assert "export type ActionData =" in after
    assert "export type ActionStreamingData = {" in after
    assert "[TType in ActionData[\"type\"]]" in after


def test_trigger_image_paths_survive_storage_roundtrip_as_real_paths():
    # Regression: the json field_serializer that rewrites paths -> URLs must NOT
    # corrupt persisted triggers (storage uses model_dump_json under the hood).
    trigger = UserInputTrigger(
        content="hi",
        user_name="U",
        image_paths=[ImageFilePath("C:/real/path/photo.png")],
    )
    restored = TriggerSerializer.from_blob(TriggerSerializer.to_blob(trigger))
    assert restored.image_paths == ["C:/real/path/photo.png"]
    assert restored.get_images() == ["C:/real/path/photo.png"]


def test_trigger_image_paths_wire_form_is_served_urls():
    trigger = UserInputTrigger(
        content="hi",
        user_name="U",
        image_paths=[ImageFilePath("C:/real/path/photo.png")],
    )
    payload = json.loads(trigger.model_dump_json())
    assert payload["image_paths"] == ["/uploaded_images/photo.png"]


def _entry_with_embedding() -> TriggerHistoryEntry:
    return TriggerHistoryEntry(
        trigger=UserInputTrigger(content="hi", user_name="U"),
        situational_context="ctx",
        embedding_vector=[0.1, 0.2, 0.3],
        entry_id="e1",
    )


def test_trigger_history_entry_omits_embedding_vector_on_wire():
    # Embeddings live in ChromaDB; they must not bloat the client wire payload.
    entry = _entry_with_embedding()
    assert "embedding_vector" not in json.loads(entry.model_dump_json())
    # Still available in-memory so ChromaDB indexing keeps working.
    assert entry.embedding_vector == [0.1, 0.2, 0.3]


def test_embedding_vector_excluded_when_entry_is_nested_in_wire_models():
    # The entry travels nested inside events/responses; exclusion must hold there too,
    # which is where a stray SerializeAsAny / dump override would regress it.
    from agent.api_types.events import (
        EventEnvelope,
        HydrationResponse,
        TriggerCompletedEvent,
    )
    from agent.api_types.timeline import (
        PaginationInfo,
        TimelineEntryTrigger,
        TimelineResponse,
    )

    entry = _entry_with_embedding()
    pagination = PaginationInfo(
        total_items=1, page_size=20, has_next=False, has_previous=False
    )

    envelope = EventEnvelope(
        event_sequence=1,
        trigger_id="t1",
        event=TriggerCompletedEvent(
            entry=entry,
            estimated_tokens=1,
            context_limit=2,
            usage_percentage=0.5,
            approaching_limit=False,
        ),
    )
    timeline = TimelineResponse(
        entries=[TimelineEntryTrigger(entry=entry)], pagination=pagination
    )
    hydration = HydrationResponse(
        entries=[TimelineEntryTrigger(entry=entry)], pagination=pagination
    )

    for model in (envelope, timeline, hydration):
        assert "embedding_vector" not in model.model_dump_json()
