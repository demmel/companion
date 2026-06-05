"""
Trigger-based history system - replaces conversation-based history with stream of consciousness approach.

Instead of back-and-forth conversation history, this tracks triggers and the agent's responses
to them, allowing for more flexible interaction patterns beyond just user messages.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.action.data.think_data import ThinkActionData
from agent.chain_of_action.trigger import Trigger, BirthTrigger
from agent.state import State, Priority, Value


def _parse_initial_state_from_think(thoughts: str) -> State:
    """
    Parse initial state from THINK action output.

    Format expected:
    Name: {name}
    Role: {role}
    Mood: {mood}
    Environment: {environment}
    Appearance: {appearance}
    Backstory: {backstory}
    Core Values:
    - {value1}
    Priorities:
    - {priority1}
    """
    lines = thoughts.split("\n")

    name = ""
    role = ""
    mood = ""
    environment = ""
    appearance = ""
    core_values: list[Value] = []
    priorities: list[str] = []

    current_section: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Name:"):
            name = line[5:].strip()
        elif line.startswith("Role:"):
            role = line[5:].strip()
        elif line.startswith("Mood:"):
            mood = line[5:].strip()
        elif line.startswith("Environment:"):
            environment = line[12:].strip()
        elif line.startswith("Appearance:"):
            appearance = line[11:].strip()
        elif line.startswith("Backstory:"):
            current_section = "backstory"
        elif line == "Core Values:":
            current_section = "values"
        elif line == "Priorities:":
            current_section = "priorities"
        elif line.startswith("- "):
            item = line[2:].strip()
            if current_section == "values":
                core_values.append(Value(content=item))
            elif current_section == "priorities":
                priorities.append(item)

    priority_objects = [
        Priority(id=f"p{i}", content=content) for i, content in enumerate(priorities, 1)
    ]

    return State(
        name=name,
        role=role,
        current_mood=mood,
        mood_intensity="medium",
        current_appearance=appearance,
        current_environment=environment,
        core_values=core_values,
        current_priorities=priority_objects,
        next_priority_id=len(priority_objects) + 1,
    )


class SummaryRecord(BaseModel):
    """Record of a summary and where it should appear in the UI"""

    summary_text: str
    insert_at_index: int  # Where this summary appears in the UI chronologically
    created_at: datetime = Field(default_factory=datetime.now)


class TriggerHistoryEntry(BaseModel):
    """Single entry in trigger-based history - a trigger and agent's response to it"""

    trigger: Trigger
    actions_taken: List[ActionData] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    end_timestamp: Optional[datetime] = Field(default=None)
    entry_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    situational_context: str
    compressed_summary: Optional[str] = Field(default=None)
    # Excluded from serialization: embeddings live in ChromaDB and are reattached
    # in-memory on load; serializing them would bloat the client wire payload and
    # dag.json without being read back from either.
    embedding_vector: Optional[List[float]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def populate_birth_trigger_initial_state(self) -> "TriggerHistoryEntry":
        """Populate initial_state on BirthTrigger from THINK action if not set."""
        if not isinstance(self.trigger, BirthTrigger):
            return self

        if self.trigger.initial_state is not None:
            return self

        # Find THINK action and parse initial state
        for action in self.actions_taken:
            if isinstance(action, ThinkActionData) and action.result.type == "success":
                thoughts = action.result.content.thoughts
                self.trigger.initial_state = _parse_initial_state_from_think(thoughts)
                break

        return self
