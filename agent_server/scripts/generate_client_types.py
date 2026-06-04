"""Generate the React client's TypeScript wire types from backend Pydantic models."""

from __future__ import annotations

import inspect
from collections import deque
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent.api_types import api as api_types
from agent.api_types import events as event_types
from agent.api_types import timeline as timeline_types
from agent.chain_of_action.action.action_data import _ACTION_DATA_TYPES
from agent.chain_of_action.action.base_action_data import ActionFailureResult
from agent.chain_of_action.trigger import BirthTrigger, UserInputTrigger, WakeupTrigger
from agent.chain_of_action.trigger_history_entry import SummaryRecord, TriggerHistoryEntry


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "client" / "src" / "types.ts"


class TypeEmitter:
    def __init__(self) -> None:
        self.queue: deque[type[BaseModel]] = deque()
        self.seen: set[type[BaseModel]] = set()
        self.interfaces: list[str] = []

    def enqueue(self, model: type[BaseModel]) -> None:
        if model in self.seen or model in _ACTION_DATA_TYPES:
            return
        self.seen.add(model)
        self.queue.append(model)

    def ts_type(self, annotation: Any) -> str:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if annotation is Any:
            return "unknown"
        if annotation in (str, Path):
            return "string"
        if annotation in (int, float):
            return "number"
        if annotation is bool:
            return "boolean"
        if annotation in (datetime, date):
            return "string"
        if annotation is NoneType:
            return "null"
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            return " | ".join(f'"{item.value}"' for item in annotation)
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            self.enqueue(annotation)
            return annotation.__name__

        if origin is list or origin is tuple:
            item_type = self.ts_type(args[0]) if args else "unknown"
            if " | " in item_type:
                item_type = f"({item_type})"
            return f"{item_type}[]"
        if origin is dict:
            value_type = self.ts_type(args[1]) if len(args) > 1 else "unknown"
            return f"Record<string, {value_type}>"
        if origin is type:
            return "unknown"
        if origin is UnionType or origin is not None and str(origin) == "typing.Union":
            parts = [self.ts_type(arg) for arg in args]
            return " | ".join(dict.fromkeys(parts))
        if str(origin) == "typing.Literal":
            return " | ".join(self.literal_value(arg) for arg in args)
        if getattr(annotation, "__name__", "") == "ImageFilePath":
            return "string"

        return "unknown"

    def literal_value(self, value: Any) -> str:
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            return f'"{value}"'
        if value is None:
            return "null"
        return repr(value).lower()

    def emit_model(self, model: type[BaseModel]) -> None:
        lines = [f"export interface {model.__name__} {{"]
        for name, field in model.model_fields.items():
            # Fields excluded from serialization are not on the wire, so they must
            # not appear in the generated client types.
            if field.exclude:
                continue
            if model is TriggerHistoryEntry and name == "actions_taken":
                ts_type = "Action[]"
            else:
                ts_type = self.ts_type(field.annotation)
            if name == "type" and field.default is not PydanticUndefined:
                ts_type = self.literal_value(field.default)
            lines.append(f"  {name}: {ts_type};")
        lines.append("}")
        self.interfaces.append("\n".join(lines))

    def emit_all(self) -> str:
        while self.queue:
            model = self.queue.popleft()
            self.emit_model(model)
        return "\n\n".join(self.interfaces)


def action_result_type_name(action_data_cls: type[BaseModel]) -> str:
    result_annotation = action_data_cls.model_fields["result"].annotation
    output_cls = get_args(result_annotation)[0]
    return output_cls.__name__


def action_alias(action_data_cls: type[BaseModel]) -> str:
    return action_data_cls.__name__.removesuffix("Data")


def main() -> None:
    emitter = TypeEmitter()

    for action_data_cls in _ACTION_DATA_TYPES:
        emitter.enqueue(action_data_cls.model_fields["input"].annotation)
        emitter.enqueue(get_args(action_data_cls.model_fields["result"].annotation)[0])

    for model in (UserInputTrigger, WakeupTrigger, BirthTrigger):
        emitter.enqueue(model)

    for model in (
        SummaryRecord,
        TriggerHistoryEntry,
        timeline_types.TimelineEntryTrigger,
        timeline_types.TimelineEntrySummary,
        timeline_types.PaginationInfo,
        timeline_types.TimelineResponse,
        event_types.TriggerStartedEvent,
        event_types.ActionStartedEvent,
        event_types.ActionProgressEvent,
        event_types.ActionCompletedEvent,
        event_types.TriggerCompletedEvent,
        event_types.AgentErrorEvent,
        event_types.SummarizationStartedEvent,
        event_types.SummarizationFinishedEvent,
        event_types.EventEnvelope,
        event_types.HydrationResponse,
    ):
        emitter.enqueue(model)

    for _, model in inspect.getmembers(api_types, inspect.isclass):
        if issubclass(model, BaseModel) and model.__module__ == api_types.__name__:
            emitter.enqueue(model)

    header = """// Generated by agent_server/scripts/generate_client_types.py.
// Do not edit by hand.
"""

    action_results = """export interface ActionSuccessResult<TContent> {
  type: "success";
  content: TContent;
}

export interface ActionFailureResult {
  type: "failure";
  error: string;
}

export interface ActionStreamingResult {
  type: "streaming";
  result: string;
}

export type WireActionResult<TContent> =
  | ActionSuccessResult<TContent>
  | ActionFailureResult;

export type ActionResult<TContent> =
  | WireActionResult<TContent>
  | ActionStreamingResult;

export interface ContextInfo {
  estimated_tokens: number;
  context_limit: number;
  usage_percentage: number;
  conversation_messages: number;
  approaching_limit: boolean;
}
"""

    action_interfaces = []
    action_aliases = []
    action_union_members = []
    for action_data_cls in _ACTION_DATA_TYPES:
        type_value = action_data_cls.model_fields["type"].default.value
        input_name = action_data_cls.model_fields["input"].annotation.__name__
        output_name = action_result_type_name(action_data_cls)
        data_name = action_data_cls.__name__
        action_interfaces.append(
            f"""export interface {data_name} {{
  type: "{type_value}";
  reasoning: string;
  input: {input_name};
  result: WireActionResult<{output_name}>;
  duration_ms: number;
  start_timestamp: string;
}}"""
        )
        alias_name = action_alias(action_data_cls)
        action_aliases.append(
            f'export type {alias_name} = Extract<Action, {{ type: "{type_value}" }}>;'
        )
        action_union_members.append(data_name)

    action_data_union = " | ".join(action_union_members)
    unions = f"""export type ActionData = {action_data_union};
export type ActionStreamingData = {{
  [TType in ActionData["type"]]: {{
  type: TType;
  reasoning: string;
  result: ActionStreamingResult;
  duration_ms: number;
  start_timestamp: string;
  }}
}}[ActionData["type"]];
export type Action = ActionData | ActionStreamingData;
export type Trigger = UserInputTrigger | WakeupTrigger | BirthTrigger;
export type TimelineEntry = TimelineEntryTrigger | TimelineEntrySummary;
export type AgentEvent =
  | TriggerStartedEvent
  | ActionStartedEvent
  | ActionProgressEvent
  | ActionCompletedEvent
  | TriggerCompletedEvent
  | AgentErrorEvent
  | SummarizationStartedEvent
  | SummarizationFinishedEvent;
export type AgentServerEvent = HydrationResponse | EventEnvelope;
export type InstalledOllamaModel = InstalledOllamaModelResponse;
"""

    content = "\n\n".join(
        [
            header.rstrip(),
            action_results,
            emitter.emit_all(),
            "\n\n".join(action_interfaces),
            unions,
            "\n".join(action_aliases),
        ]
    )
    OUTPUT.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
