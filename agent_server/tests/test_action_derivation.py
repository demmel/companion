"""
Tests for the derived action registry / ActionData machinery.

These cover the mechanisms introduced when the per-action wiring was replaced with
derivation from the @register_action registry:
  - the action registry is complete and authoritative,
  - the @register_action decorator registers and returns the class unchanged,
  - get_input_type() is derived correctly from the generic parameter,
  - the ActionData union and the type->class constructor map are derived and in parity,
  - the `type` discriminator strings are stable (persisted blobs depend on them),
  - every action type's ActionData round-trips through the storage serializer,
  - isinstance narrowing on the concrete classes still discriminates correctly,
  - ActionRegistry availability honours can_perform.
"""

import json
import typing
import zlib
from datetime import datetime

import pytest
from pydantic import BaseModel

from agent.state import create_default_agent_state
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.action.base_action import (
    _ACTION_REGISTRY,
    BaseAction,
    register_action,
)
from agent.chain_of_action.action.base_action_data import (
    _ACTION_DATA_TYPES,
    ActionFailureResult,
    ActionSuccessResult,
    BaseActionData,
)
from agent.chain_of_action.action.action_data import (
    ActionData,
    ACTION_DATA_UNION,
    _ACTION_DATA_CONSTRUCTORS,
    create_action_data,
    create_result_summary,
)
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.chain_of_action.action_registry import ActionRegistry
from agent.storage.serializers import ActionSerializer, _ACTION_DATA_ADAPTER

from agent.chain_of_action.action.actions.think_action import (
    ThinkActionData,
    ThinkInput,
    ThinkOutput,
)
from agent.chain_of_action.action.actions.speak_action import (
    SpeakActionData,
    SpeakInput,
    SpeakOutput,
)
from agent.chain_of_action.action.actions.wait_action import WaitInput, WaitOutput
from agent.chain_of_action.action.actions.update_mood_action import (
    UpdateMoodInput,
    UpdateMoodOutput,
)
from agent.chain_of_action.action.actions.visual_actions import (
    UpdateAppearanceInput,
    UpdateEnvironmentInput,
)
from agent.chain_of_action.action.actions.fetch_url_action import FetchUrlInput
from agent.chain_of_action.action.actions.search_web_action import (
    SearchWebInput,
    SearchWebOutput,
    SearchResult,
)
from agent.chain_of_action.action.actions.priority_actions import (
    AddPriorityInput,
    RemovePriorityInput,
)
from agent.chain_of_action.action.actions.evaluate_priorities_action import (
    EvaluatePrioritiesInput,
)
from agent.chain_of_action.action.actions.creative_inspiration_action import (
    CreativeInspirationInput,
)
from agent.chain_of_action.action.actions.remember_action import RememberInput
from agent.memory.queries import MemoryQuery, QueryType


# Every ActionType, with the exact discriminator string persisted in existing DBs.
# This map is intentionally hard-coded: it is the stability contract for stored blobs.
EXPECTED_DISCRIMINATORS: dict[ActionType, str] = {
    ActionType.THINK: "think",
    ActionType.UPDATE_MOOD: "update_mood",
    ActionType.UPDATE_APPEARANCE: "update_appearance",
    ActionType.UPDATE_ENVIRONMENT: "update_environment",
    ActionType.ADD_PRIORITY: "add_priority",
    ActionType.REMOVE_PRIORITY: "remove_priority",
    ActionType.EVALUATE_PRIORITIES: "evaluate_priorities",
    ActionType.SPEAK: "speak",
    ActionType.FETCH_URL: "fetch_url",
    ActionType.SEARCH_WEB: "search_web",
    ActionType.WAIT: "wait",
    ActionType.GET_CREATIVE_INSPIRATION: "get_creative_inspiration",
    ActionType.REMEMBER: "remember",
}

# A valid input instance for every action type, so we can exercise serialization of
# every variant (failure results need only an input; success outputs are added below).
INPUTS_BY_TYPE: dict[ActionType, BaseModel] = {
    ActionType.THINK: ThinkInput(focus="what to do"),
    ActionType.UPDATE_MOOD: UpdateMoodInput(
        reason="good news", new_mood="happy", intensity="high"
    ),
    ActionType.UPDATE_APPEARANCE: UpdateAppearanceInput(
        reason="feeling fresh", change_description="put on a red scarf"
    ),
    ActionType.UPDATE_ENVIRONMENT: UpdateEnvironmentInput(
        reason="cozier", change_description="dim the lights"
    ),
    ActionType.ADD_PRIORITY: AddPriorityInput(
        reason="matters", priority_content="learn guitar"
    ),
    ActionType.REMOVE_PRIORITY: RemovePriorityInput(reason="done", priority_id="p1"),
    ActionType.EVALUATE_PRIORITIES: EvaluatePrioritiesInput(focus="re-align"),
    ActionType.SPEAK: SpeakInput(intent="greet", tone="warm"),
    ActionType.FETCH_URL: FetchUrlInput(url="https://example.com", looking_for="facts"),
    ActionType.SEARCH_WEB: SearchWebInput(purpose="learn", query="python asyncio"),
    ActionType.WAIT: WaitInput(reason="done for now"),
    ActionType.GET_CREATIVE_INSPIRATION: CreativeInspirationInput(),
    ActionType.REMEMBER: RememberInput(
        reason="ground myself",
        queries=[
            MemoryQuery(
                reasoning="need past context",
                query_type=QueryType.RELATIONSHIP,
                query_text="what David does for work",
                importance=0.9,
            )
        ],
    ),
}


def _make(action_type: ActionType, input_model: BaseModel, result) -> BaseActionData:
    return create_action_data(
        type=action_type,
        reasoning="because",
        input=input_model,
        result=result,
        duration_ms=12.5,
        start_timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )


class TestRegistryDerivation:
    def test_registry_covers_every_action_type(self) -> None:
        assert set(_ACTION_REGISTRY.keys()) == set(ActionType)

    def test_registry_values_are_base_action_subclasses(self) -> None:
        for cls in _ACTION_REGISTRY.values():
            assert issubclass(cls, BaseAction)

    def test_action_registry_exposes_all_actions(self) -> None:
        registry = ActionRegistry()
        assert set(registry.get_available_actions()) == set(ActionType)

    def test_create_action_returns_instance_for_every_type(self) -> None:
        registry = ActionRegistry(enable_image_generation=False)
        for action_type in ActionType:
            assert isinstance(registry.create_action(action_type), BaseAction)

    def test_action_descriptions_present_for_every_type(self) -> None:
        registry = ActionRegistry()
        descriptions = registry.get_action_descriptions()
        assert set(descriptions.keys()) == set(ActionType)
        assert all(isinstance(d, str) and d for d in descriptions.values())


class TestRegisterActionDecorator:
    def test_decorator_returns_class_unchanged(self) -> None:
        # The 12 real actions are reachable as their own classes via the registry,
        # which is only possible if the decorator returned the class it decorated.
        from agent.chain_of_action.action.actions.think_action import ThinkAction

        assert _ACTION_REGISTRY[ActionType.THINK] is ThinkAction

    def test_decorator_registers_and_is_identity(self) -> None:
        original = dict(_ACTION_REGISTRY)
        try:

            @register_action(ActionType.THINK)
            class Sentinel(BaseAction[ThinkInput, ThinkOutput]):
                @classmethod
                def get_action_description(cls) -> str:
                    return "sentinel"

                def execute(self, action_input, context, state, llm, progress_callback):
                    raise NotImplementedError

            assert _ACTION_REGISTRY[ActionType.THINK] is Sentinel
        finally:
            _ACTION_REGISTRY.clear()
            _ACTION_REGISTRY.update(original)
        assert _ACTION_REGISTRY[ActionType.THINK] is not None


class TestGetInputTypeDerivation:
    def test_get_input_type_is_a_model_for_every_action(self) -> None:
        for cls in _ACTION_REGISTRY.values():
            input_type = cls.get_input_type()
            assert isinstance(input_type, type) and issubclass(input_type, BaseModel)

    def test_get_input_type_matches_generic_argument(self) -> None:
        for cls in _ACTION_REGISTRY.values():
            declared_input = None
            for base in cls.__orig_bases__:
                origin = typing.get_origin(base)
                if origin is not None and issubclass(origin, BaseAction):
                    declared_input = typing.get_args(base)[0]
                    break
            assert declared_input is not None
            assert cls.get_input_type() is declared_input

    def test_get_input_type_raises_without_generic_parameter(self) -> None:
        class Unparameterized(BaseAction):  # type: ignore[type-arg]
            @classmethod
            def get_action_description(cls) -> str:
                return "x"

            def execute(self, action_input, context, state, llm, progress_callback):
                raise NotImplementedError

        with pytest.raises(TypeError):
            Unparameterized.get_input_type()


class TestActionDataUnionDerivation:
    def test_action_data_annotation_is_runtime_union(self) -> None:
        assert typing.get_args(ActionData) == typing.get_args(ACTION_DATA_UNION)

    def test_union_has_one_member_per_action_type(self) -> None:
        assert len(typing.get_args(ACTION_DATA_UNION)) == len(ActionType)
        assert len(_ACTION_DATA_TYPES) == len(ActionType)

    def test_constructor_map_parity(self) -> None:
        assert set(_ACTION_DATA_CONSTRUCTORS.keys()) == set(ActionType)

    def test_every_member_is_a_base_action_data_subclass(self) -> None:
        for member in typing.get_args(ACTION_DATA_UNION):
            assert issubclass(member, BaseActionData)

    def test_union_members_are_distinct(self) -> None:
        members = typing.get_args(ACTION_DATA_UNION)
        assert len(set(members)) == len(members)

    def test_discriminators_are_stable(self) -> None:
        for action_type, cls in _ACTION_DATA_CONSTRUCTORS.items():
            default = cls.model_fields["type"].default
            assert default == action_type
            assert default.value == EXPECTED_DISCRIMINATORS[action_type]

    def test_expected_discriminators_cover_all_types(self) -> None:
        assert set(EXPECTED_DISCRIMINATORS.keys()) == set(ActionType)


class TestCreateActionData:
    @pytest.mark.parametrize("action_type", list(ActionType))
    def test_returns_correct_concrete_class(self, action_type: ActionType) -> None:
        data = _make(
            action_type, INPUTS_BY_TYPE[action_type], ActionFailureResult(error="e")
        )
        assert type(data) is _ACTION_DATA_CONSTRUCTORS[action_type]
        assert data.type == action_type

    @pytest.mark.parametrize("action_type", list(ActionType))
    def test_isinstance_discriminates(self, action_type: ActionType) -> None:
        data = _make(
            action_type, INPUTS_BY_TYPE[action_type], ActionFailureResult(error="e")
        )
        assert isinstance(data, _ACTION_DATA_CONSTRUCTORS[action_type])
        for other_type, other_cls in _ACTION_DATA_CONSTRUCTORS.items():
            if other_type != action_type:
                assert not isinstance(data, other_cls)

    def test_result_summary_uses_error_on_failure(self) -> None:
        data = _make(
            ActionType.THINK,
            INPUTS_BY_TYPE[ActionType.THINK],
            ActionFailureResult(error="boom"),
        )
        assert create_result_summary(data) == "boom"


class TestSerializationRoundTrip:
    @pytest.mark.parametrize("action_type", list(ActionType))
    def test_failure_roundtrips_for_every_type(self, action_type: ActionType) -> None:
        action = _make(
            action_type, INPUTS_BY_TYPE[action_type], ActionFailureResult(error="x")
        )
        row = ActionSerializer.to_row(action, trigger_entry_id="t1", sequence=0)
        assert row.action_type == EXPECTED_DISCRIMINATORS[action_type]
        assert row.result_type == "failure"

        restored = ActionSerializer.from_row(row)
        assert type(restored) is type(action)
        assert restored == action
        assert restored.type == action_type

    def test_success_roundtrip_simple_output(self) -> None:
        action = _make(
            ActionType.THINK,
            ThinkInput(focus="x"),
            ActionSuccessResult(content=ThinkOutput(thoughts="deep thought")),
        )
        restored = ActionSerializer.from_row(ActionSerializer.to_row(action, "t", 0))
        assert isinstance(restored, ThinkActionData)
        assert restored.result.type == "success"
        assert restored.result.content.thoughts == "deep thought"

    def test_success_roundtrip_nested_output(self) -> None:
        action = _make(
            ActionType.SEARCH_WEB,
            SearchWebInput(purpose="learn", query="q"),
            ActionSuccessResult(
                content=SearchWebOutput(
                    query_used="q",
                    search_results=[
                        SearchResult(title="T", url="https://x", snippet="s")
                    ],
                    total_results_found=1,
                )
            ),
        )
        restored = ActionSerializer.from_row(ActionSerializer.to_row(action, "t", 0))
        assert restored == action
        assert restored.result.type == "success"
        assert restored.result.content.search_results[0].url == "https://x"

    def test_speak_success_roundtrip(self) -> None:
        action = _make(
            ActionType.SPEAK,
            SpeakInput(intent="greet", tone="warm"),
            ActionSuccessResult(content=SpeakOutput(response="hi")),
        )
        restored = ActionSerializer.from_row(ActionSerializer.to_row(action, "t", 0))
        assert isinstance(restored, SpeakActionData)
        assert restored.input.tone == "warm"
        assert restored.result.content.response == "hi"


class TestBackwardCompatibility:
    def test_blob_with_extra_unknown_field_is_ignored(self) -> None:
        # Older blobs may carry fields no longer modelled; extra="ignore" must drop them.
        action = _make(
            ActionType.UPDATE_MOOD,
            UpdateMoodInput(reason="r", new_mood="happy", intensity="high"),
            ActionSuccessResult(
                content=UpdateMoodOutput(
                    old_mood="neutral",
                    old_intensity="medium",
                    new_mood="happy",
                    new_intensity="high",
                    reason="r",
                )
            ),
        )
        payload = json.loads(action.model_dump_json())
        payload["legacy_extra_field"] = "should be ignored"
        restored = _ACTION_DATA_ADAPTER.validate_json(
            json.dumps(payload).encode("utf-8")
        )
        assert restored.type == ActionType.UPDATE_MOOD
        assert restored.result.content.new_mood == "happy"

    def test_discriminator_in_blob_selects_class(self) -> None:
        # The `type` field in the blob is what drives concrete-class selection.
        for action_type, cls in _ACTION_DATA_CONSTRUCTORS.items():
            action = _make(
                action_type, INPUTS_BY_TYPE[action_type], ActionFailureResult(error="e")
            )
            blob = action.model_dump_json().encode("utf-8")
            restored = _ACTION_DATA_ADAPTER.validate_json(blob)
            assert type(restored) is cls

    def test_trigger_history_entry_json_restores_concrete_action_data(self) -> None:
        action = _make(
            ActionType.THINK,
            ThinkInput(focus="x"),
            ActionSuccessResult(content=ThinkOutput(thoughts="deep thought")),
        )
        entry = TriggerHistoryEntry(
            trigger=UserInputTrigger(content="hello", user_name="User"),
            actions_taken=[action],
            situational_context="User said hello",
        )

        restored = TriggerHistoryEntry.model_validate_json(entry.model_dump_json())

        assert isinstance(restored.actions_taken[0], ThinkActionData)
        assert restored.actions_taken[0].result.type == "success"
        assert restored.actions_taken[0].result.content.thoughts == "deep thought"


class TestRegistryCanPerform:
    def test_remove_priority_unavailable_without_priorities(self) -> None:
        registry = ActionRegistry()
        state = create_default_agent_state()
        state.current_priorities = []
        available = registry.get_available_actions_for_state(state)
        assert ActionType.REMOVE_PRIORITY not in available
        # Non-conditional actions remain available.
        assert ActionType.THINK in available
