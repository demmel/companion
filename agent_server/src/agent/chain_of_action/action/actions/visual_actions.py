"""
Visual state update actions - appearance and environment updates.
"""

from __future__ import annotations

import logging
from typing import Type, Optional

from agent.chain_of_action.context import ExecutionContext
from agent.types import ImageGenerationToolContent, ToolCallError

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)

from agent.state import State
from agent.llm import LLM, SupportedModel, ImagesInput
from agent.structured_llm import direct_structured_llm_call

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Shared input/output models
class UpdateAppearanceInput(BaseModel):
    """Input for UPDATE_APPEARANCE action"""

    reason: str = Field(description="Why I'm changing my appearance")
    change_description: str = Field(
        description="What specific aspects of my appearance should change and how. These should be specific and detailed."
    )


class UpdateEnvironmentInput(BaseModel):
    """Input for UPDATE_ENVIRONMENT action"""

    reason: str = Field(description="Why I'm changing my environment")
    change_description: str = Field(
        description="What specific aspects of my environment should change and how. These should be specific and detailed."
    )


class UpdateAppearanceOutput(ActionOutput):
    """Output for UPDATE_APPEARANCE action"""

    image_description: str
    old_appearance: str
    new_appearance: str
    reason: str
    image_result: ImageGenerationToolContent

    def result_summary(self) -> str:
        return f"Appearance updated: {self.new_appearance} (reason: {self.reason})"


class UpdateEnvironmentOutput(ActionOutput):
    """Output for UPDATE_ENVIRONMENT action"""

    image_description: str
    old_environment: str
    new_environment: str
    reason: str
    image_result: ImageGenerationToolContent

    def result_summary(self) -> str:
        return f"Environment updated: {self.new_environment} (reason: {self.reason})"


# Shared LLM response models
class AppearanceUpdate(BaseModel):
    """Updated appearance description"""

    updated_appearance: str = Field(
        description="Updated appearance description in first-person that builds on current state. MUST start with 'I' and follow this exact order: 1) Pose/posture 2) Facial expression 3) Hair/body details 4) Clothing/accessories. PRIORITIZE emotional expression over clothing. Example: 'I'm leaning forward with a gentle smile, my hair falling softly around my shoulders, wearing a simple blue dress.' Always start with current appearance and modify only what was requested to change."
    )


class EnvironmentUpdate(BaseModel):
    """Updated environment description"""

    updated_environment: str = Field(
        description="Updated environment description in first-person that builds on current state. MUST start with 'I' and describe the setting around me. Example: 'I'm in a cozy coffee shop with warm lighting and soft jazz playing in the background.' Always start with current environment and modify only what was requested to change."
    )


# Shared utility functions
def _build_image_description(
    appearance: str,
    environment: str,
    agent_name: str,
    llm: LLM,
    model: SupportedModel,
    images: ImagesInput = None,
) -> str:
    """Use LLM to convert first-person descriptions and combine into image generation prompt"""

    # Handle empty descriptions
    if not appearance and not environment:
        return f"{agent_name}"
    elif not appearance:
        appearance = f"I am {agent_name}"
    elif not environment:
        environment = "I'm in a simple setting"

    prompt = f"""Create an image generation description by converting these first-person descriptions to third-person and combining them naturally:

Character name: {agent_name}
Appearance: "{appearance}"
Environment: "{environment}"

Convert to third-person and combine into a coherent image description suitable for AI image generation. Focus on visual details that can be rendered. Keep it concise but descriptive.

Image description:"""

    response = llm.generate_complete(
        model, prompt, caller="build_image_description", images=images
    )
    return response.strip()


def _generate_image_if_enabled(
    image_description: str,
    enable_image_generation: bool,
    llm: LLM,
    model: SupportedModel,
    progress_callback,
) -> ImageGenerationToolContent:
    """Generate image if enabled, otherwise return placeholder"""

    if enable_image_generation:
        from agent.image_generation import get_shared_image_generator

        image_generator = get_shared_image_generator()

        def image_progress_callback(progress_data):
            progress_callback(progress_data)

        image_tool_result = image_generator.generate_image_for_action(
            image_description, llm, model, image_progress_callback
        )

        if isinstance(image_tool_result, ToolCallError):
            raise Exception(f"Image generation failed: {image_tool_result.error}")

        assert isinstance(
            image_tool_result.content, ImageGenerationToolContent
        ), "Image generation must return ImageGenerationToolContent"

        return image_tool_result.content
    else:
        return ImageGenerationToolContent(
            prompt=image_description,
            image_path="",
            image_url="/placeholder-image.png",
            width=512,
            height=512,
            num_inference_steps=0,
            guidance_scale=0.0,
        )


# Action implementations
class UpdateAppearanceAction(BaseAction[UpdateAppearanceInput, UpdateAppearanceOutput]):
    """Update the agent's appearance and generate a new image"""

    action_type = ActionType.UPDATE_APPEARANCE

    def __init__(self, enable_image_generation: bool = True):
        self.enable_image_generation = enable_image_generation

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my appearance (posture, expression, hair, clothing) and generate a new image"

    @classmethod
    def get_input_type(cls) -> Type[UpdateAppearanceInput]:
        return UpdateAppearanceInput

    def execute(
        self,
        action_input: UpdateAppearanceInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[UpdateAppearanceOutput]:
        return _execute_appearance_update(
            action_input,
            context,
            state,
            llm,
            model,
            progress_callback,
            self.enable_image_generation,
        )


class UpdateEnvironmentAction(
    BaseAction[UpdateEnvironmentInput, UpdateEnvironmentOutput]
):
    """Update the agent's environment and generate a new image"""

    action_type = ActionType.UPDATE_ENVIRONMENT

    def __init__(self, enable_image_generation: bool = True):
        self.enable_image_generation = enable_image_generation

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my environment (setting, location, surroundings) and generate a new image"

    @classmethod
    def get_input_type(cls) -> Type[UpdateEnvironmentInput]:
        return UpdateEnvironmentInput

    def execute(
        self,
        action_input: UpdateEnvironmentInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[UpdateEnvironmentOutput]:
        return _execute_environment_update(
            action_input,
            context,
            state,
            llm,
            model,
            progress_callback,
            self.enable_image_generation,
        )


# Shared execution logic
def _execute_appearance_update(
    action_input: UpdateAppearanceInput,
    context: ExecutionContext,
    state: State,
    llm: LLM,
    model: SupportedModel,
    progress_callback,
    enable_image_generation: bool = True,
) -> ActionResult[UpdateAppearanceOutput]:
    """Shared logic for appearance updates"""
    from agent.chain_of_action.prompts import (
        format_section,
        format_action_sequence_status,
    )

    logger.debug("=== UPDATE_APPEARANCE ACTION ===")
    logger.debug(f"APPEARANCE CHANGE: {action_input.change_description}")
    logger.debug(f"REASON: {action_input.reason}")

    old_appearance = state.current_appearance

    # Show full action sequence with status indicators
    action_sequence = format_action_sequence_status(
        context.completed_actions,
        context.planned_actions,
        context.current_action_index,
    )
    actions_taken_section = format_section("MY ACTION PLAN", action_sequence)

    appearance_update_section = format_section(
        "PROPOSED APPEARANCE UPDATE",
        f"""
Current appearance: "{old_appearance}"
Requested change: "{action_input.change_description}"
Reason for change: "{action_input.reason}"
""",
    )

    merge_prompt = f"""I am {state.name}, {state.role}, and I need to update my appearance based on a specific change request. Please merge the requested changes with my current appearance to create an absolute description.

{actions_taken_section}

{appearance_update_section}

I should provide an updated appearance description that:
1. Builds on my current appearance
2. Incorporates the requested changes as absolute states (not comparative language like "more" or "leaning forward than before")
3. Follows the format: pose/posture, facial expression, hair/body details, clothing/accessories
4. Uses first-person perspective starting with "I"
5. Incorporates any actions I've taken prior to this update that should alter my appearance or override the requested changes
6. Use explicit language to describe my appearance instead of vague terms. (e.g. Bad: "Professional outfit", Good: "Wearing a tailored navy blazer over a crisp white shirt", Bad: "vibrant colors", Good: "a bright red scarf, a green dress with yellow flowers")
7. IMPORTANT: Convert any comparative language to absolute descriptions (e.g., "leaning more forward" becomes "leaning forward", "feeling happier" becomes "smiling warmly")
8. CRITICAL: I must change at least one concrete aspect of my appearance from my current state. Avoid generic descriptions and ensure the new appearance is meaningfully different from the current one.

The result should be a natural evolution of my current appearance with the requested modifications expressed as absolute states."""

    try:
        # Get updated appearance using structured LLM call
        appearance_update = direct_structured_llm_call(
            prompt=merge_prompt,
            response_model=AppearanceUpdate,
            model=model,
            llm=llm,
            caller="update_appearance_action",
        )

        # Update state with new appearance
        state.current_appearance = appearance_update.updated_appearance

        # Get images from trigger
        trigger_images = context.trigger.get_images()

        # Generate image with updated appearance
        image_description = _build_image_description(
            state.current_appearance,
            state.current_environment,
            state.name,
            llm,
            model,
            trigger_images,
        )

        logger.debug(f"Generated image description: {image_description}")

        # Generate image
        image_result = _generate_image_if_enabled(
            image_description, enable_image_generation, llm, model, progress_callback
        )

        result = ActionSuccessResult(
            content=UpdateAppearanceOutput(
                image_description=image_description,
                old_appearance=old_appearance,
                new_appearance=appearance_update.updated_appearance,
                reason=action_input.reason,
                image_result=image_result,
            )
        )

        return result

    except Exception as e:
        logger.error(f"Failed to update appearance: {e}")
        return ActionFailureResult(error=f"Failed to update appearance: {str(e)}")


def _execute_environment_update(
    action_input: UpdateEnvironmentInput,
    context: ExecutionContext,
    state: State,
    llm: LLM,
    model: SupportedModel,
    progress_callback,
    enable_image_generation: bool = True,
) -> ActionResult[UpdateEnvironmentOutput]:
    """Shared logic for environment updates"""
    from agent.chain_of_action.prompts import (
        format_section,
        format_action_sequence_status,
    )

    logger.debug("=== UPDATE_ENVIRONMENT ACTION ===")
    logger.debug(f"ENVIRONMENT CHANGE: {action_input.change_description}")
    logger.debug(f"REASON: {action_input.reason}")

    old_environment = state.current_environment

    # Show full action sequence with status indicators
    action_sequence = format_action_sequence_status(
        context.completed_actions,
        context.planned_actions,
        context.current_action_index,
    )
    actions_taken_section = format_section("MY ACTION PLAN", action_sequence)

    environment_update_section = format_section(
        "PROPOSED ENVIRONMENT UPDATE",
        f"""
Current environment: "{old_environment}"
Requested change: "{action_input.change_description}"
Reason for change: "{action_input.reason}"
""",
    )

    merge_prompt = f"""I am {state.name}, {state.role}, and I need to update my environment based on a specific change request. Please merge the requested changes with my current environment to create an absolute description.

{actions_taken_section}

{environment_update_section}

I should provide an updated environment description that:
1. Builds on my current environment
2. Incorporates the requested changes as absolute states (not comparative language like "more" or "different than before")
3. Uses first-person perspective starting with "I"
4. Incorporates any actions I've taken prior to this update that should alter my environment or override the requested changes
5. Use explicit language to describe my environment instead of vague terms. (e.g. Bad: "Nice place", Good: "cozy coffee shop with warm wood furniture and soft lighting")
6. IMPORTANT: Convert any comparative language to absolute descriptions (e.g., "brighter lighting" becomes "bright lighting", "more natural" becomes "natural surroundings")
7. CRITICAL: I must change at least one concrete aspect of my environment from my current state. Avoid generic descriptions and ensure the new environment is meaningfully different from the current one.

The result should be a natural evolution of my current environment with the requested modifications expressed as absolute states."""

    try:
        # Get updated environment using structured LLM call
        environment_update = direct_structured_llm_call(
            prompt=merge_prompt,
            response_model=EnvironmentUpdate,
            model=model,
            llm=llm,
            caller="update_environment_action",
        )

        # Update state with new environment
        state.current_environment = environment_update.updated_environment

        # Get images from trigger
        trigger_images = context.trigger.get_images()

        # Generate image with updated environment
        image_description = _build_image_description(
            state.current_appearance,
            state.current_environment,
            state.name,
            llm,
            model,
            trigger_images,
        )

        logger.debug(f"Generated image description: {image_description}")

        # Generate image
        image_result = _generate_image_if_enabled(
            image_description, enable_image_generation, llm, model, progress_callback
        )

        result = ActionSuccessResult(
            content=UpdateEnvironmentOutput(
                image_description=image_description,
                old_environment=old_environment,
                new_environment=environment_update.updated_environment,
                reason=action_input.reason,
                image_result=image_result,
            )
        )

        return result

    except Exception as e:
        logger.error(f"Failed to update environment: {e}")
        return ActionFailureResult(error=f"Failed to update environment: {str(e)}")


# Combined execution logic for batching
def execute_visual_actions_batch(
    appearance_input: Optional[UpdateAppearanceInput],
    environment_input: Optional[UpdateEnvironmentInput],
    context: ExecutionContext,
    state: State,
    llm: LLM,
    model: SupportedModel,
    progress_callback,
    enable_image_generation: bool = True,
) -> tuple[
    Optional[ActionResult[UpdateAppearanceOutput]],
    Optional[ActionResult[UpdateEnvironmentOutput]],
]:
    """Execute appearance and/or environment updates together with a single image generation"""

    logger.debug("=== BATCHED VISUAL ACTIONS ===")

    try:
        # Execute individual actions without image generation to get state updates
        appearance_result = None
        environment_result = None

        if appearance_input:
            appearance_result = _execute_appearance_update(
                appearance_input,
                context,
                state,
                llm,
                model,
                progress_callback,
                enable_image_generation=False,
            )

        if environment_input:
            environment_result = _execute_environment_update(
                environment_input,
                context,
                state,
                llm,
                model,
                progress_callback,
                enable_image_generation=False,
            )

        # Check if any action failed
        if (appearance_result and appearance_result.type == "failure") or (
            environment_result and environment_result.type == "failure"
        ):
            return appearance_result, environment_result

        # Generate single shared image with final combined state
        trigger_images = context.trigger.get_images()
        image_description = _build_image_description(
            state.current_appearance,
            state.current_environment,
            state.name,
            llm,
            model,
            trigger_images,
        )

        logger.debug(f"Generated combined image description: {image_description}")

        shared_image_result = _generate_image_if_enabled(
            image_description, enable_image_generation, llm, model, progress_callback
        )

        # Update results with shared image
        if appearance_result and appearance_result.type == "success":
            appearance_result.content.image_result = shared_image_result
            appearance_result.content.image_description = image_description

        if environment_result and environment_result.type == "success":
            environment_result.content.image_result = shared_image_result
            environment_result.content.image_description = image_description

        return appearance_result, environment_result

    except Exception as e:
        logger.error(f"Failed to execute batched visual actions: {e}")
        error_msg = f"Failed to execute visual updates: {str(e)}"

        appearance_error = (
            ActionFailureResult(error=error_msg) if appearance_input else None
        )
        environment_error = (
            ActionFailureResult(error=error_msg) if environment_input else None
        )

        return appearance_error, environment_error
