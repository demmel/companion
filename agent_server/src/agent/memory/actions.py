"""
Action types for the temporal DAG memory system.

Actions store concrete results rather than abstract criteria to ensure deterministic replay.
Each action contains the exact changes made to the memory graph or context.
"""

import uuid
from datetime import datetime
from typing import List, Union, Literal

from pydantic import BaseModel, Field

from .models import MemoryElement, MemoryEdge, ContextElement, ConfidenceLevel


class AddMemoryAction(BaseModel):
    """Action to add a memory element to the graph."""

    action_type: Literal["add_memory"] = "add_memory"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    memory: MemoryElement


class AddEdgeAction(BaseModel):
    """Action to add a connection (edge) between memories."""

    action_type: Literal["add_connection"] = "add_connection"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    edge: MemoryEdge


class UpdateConfidenceAction(BaseModel):
    """Action to update the confidence level of a specific memory."""

    action_type: Literal["update_confidence"] = "update_confidence"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_id: str
    new_confidence: ConfidenceLevel
    reason: str


class AddToContextAction(BaseModel):
    """Action to add a memory element to the working context."""

    action_type: Literal["add_to_context"] = "add_to_context"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_id: str
    initial_tokens: int
    reinforce_tokens: int = 0


class AddEdgeToContextAction(BaseModel):
    """Action to add an edge to the working context."""

    action_type: Literal["add_edge_to_context"] = "add_edge_to_context"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    edge_id: str
    should_boost_source_tokens: bool


class RemoveFromContextAction(BaseModel):
    """Action to remove specific memories and edges from working context."""

    action_type: Literal["remove_from_context"] = "remove_from_context"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_ids: List[str]
    edge_ids: List[str]  # Exact edge IDs to remove
    reason: str


class AddContainerAction(BaseModel):
    """Action to add a memory container to the graph."""

    action_type: Literal["add_container"] = "add_container"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    container_id: str  # trigger.entry_id
    element_ids: List[str]  # Memory IDs in this container
    trigger_timestamp: datetime


class ApplyTokenDecayAction(BaseModel):
    """Action to apply token decay to specific context memories."""

    action_type: Literal["apply_token_decay"] = "apply_token_decay"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    decay_amount: int = 1  # Amount to subtract from each memory's tokens


class CheckpointAction(BaseModel):
    """Action to mark a checkpoint for visualization and replay boundaries."""

    action_type: Literal["checkpoint"] = "checkpoint"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    label: str
    description: str


# Discriminated union for all action types
MemoryAction = Union[
    AddMemoryAction,
    AddEdgeAction,
    UpdateConfidenceAction,
    AddToContextAction,
    AddEdgeToContextAction,
    RemoveFromContextAction,
    AddContainerAction,
    ApplyTokenDecayAction,
    CheckpointAction,
]
