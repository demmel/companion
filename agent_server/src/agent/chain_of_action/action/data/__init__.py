"""Action data layer.

Per-action data types (``FooInput`` / ``FooOutput`` / ``FooActionData``) live here, separate
from the action *execution* classes in ``action/actions``. This keeps the data layer free of any
dependency on ``ExecutionContext`` (and therefore on the whole execution stack), so the
discriminated ``ActionData`` union can be assembled by importing this package alone — without
dragging in execution code. That separation is what lets ``agent.memory`` (via
``TriggerHistoryEntry`` → ``ActionData``) be referenced from execution-side modules without an
import cycle.

Importing this package runs every ``*_data`` module, whose ``FooActionData`` classes self-collect
into ``_ACTION_DATA_TYPES`` (see ``base_action_data``).
"""

from .think_data import ThinkInput, ThinkOutput, ThinkActionData
from .wait_data import WaitInput, WaitOutput, WaitActionData
from .speak_data import SpeakInput, SpeakOutput, SpeakActionData
from .update_mood_data import (
    UpdateMoodInput,
    UpdateMoodOutput,
    UpdateMoodActionData,
)
from .visual_data import (
    UpdateAppearanceInput,
    UpdateEnvironmentInput,
    UpdateAppearanceOutput,
    UpdateEnvironmentOutput,
    UpdateAppearanceActionData,
    UpdateEnvironmentActionData,
)
from .fetch_url_data import FetchUrlInput, FetchUrlOutput, FetchUrlActionData
from .search_web_data import (
    SearchResult,
    SearchWebInput,
    SearchWebOutput,
    SearchWebActionData,
)
from .priority_data import (
    RelativePosition,
    AddPriorityInput,
    AddPrioritySuccessOutput,
    AddPriorityDuplicateOutput,
    AddPriorityOutput,
    RemovePriorityInput,
    RemovePriorityOutput,
    AddPriorityActionData,
    RemovePriorityActionData,
)
from .evaluate_priorities_data import (
    AddPriorityOp,
    RemovePriorityOp,
    MergePrioritiesOp,
    RefinePriorityOp,
    ReorderPriorityOp,
    PriorityOperation,
    EvaluatePrioritiesInput,
    OperationResult,
    EvaluatePrioritiesOutput,
    EvaluatePrioritiesActionData,
)
from .creative_inspiration_data import (
    CreativeInspirationInput,
    CreativeInspirationOutput,
    CreativeInspirationActionData,
)
from .remember_data import RememberInput, RememberOutput, RememberActionData
