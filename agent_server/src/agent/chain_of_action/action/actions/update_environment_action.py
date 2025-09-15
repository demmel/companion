"""
UPDATE_ENVIRONMENT action implementation.
"""

from __future__ import annotations

import logging
from typing import Type

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


class UpdateEnvironmentInput(BaseModel):
    """Input for UPDATE_ENVIRONMENT action"""

    reason: str = Field(description="Why I'm changing my environment")
    change_description: str = Field(
        description="What specific aspects of my environment should change and how. These should be specific and detailed."
    )


class UpdateEnvironmentOutput(ActionOutput):
    """Output for UPDATE_ENVIRONMENT action"""

    image_description: str
    old_environment: str
    new_environment: str
    reason: str
    image_result: ImageGenerationToolContent

    def result_summary(self) -> str:
        return f"Environment updated: {self.new_environment} (reason: {self.reason})"


class EnvironmentUpdate(BaseModel):
    """Updated environment description"""

    updated_environment: str = Field(
        description="Updated environment description in first-person that builds on current state. MUST start with 'I' and describe the setting around me. Example: 'I'm in a cozy coffee shop with warm lighting and soft jazz playing in the background.' Always start with current environment and modify only what was requested to change."
    )


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

    @classmethod
    def create_result_from_image_generation(
        cls,
        image_result: ImageGenerationToolContent,
        image_description: str,
        context_given: str,
        duration_ms: float,
        new_environment: str,
        old_environment: str = "",
        success: bool = True,
        result_summary: str = "Environment updated with new image",
    ) -> ActionResult[UpdateEnvironmentOutput]:
        """Factory method to create consistent ActionResult for UPDATE_ENVIRONMENT actions"""
        from agent.chain_of_action.action.action_types import ActionType

        return ActionSuccessResult(
            content=UpdateEnvironmentOutput(
                image_description=image_description,
                old_environment=old_environment,
                new_environment=new_environment,
                reason=result_summary,
                image_result=image_result,
            )
        )

    def execute(
        self,
        action_input: UpdateEnvironmentInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[UpdateEnvironmentOutput]:
        from agent.chain_of_action.prompts import (
            format_section,
            format_actions_for_diary,
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

        # Use LLM to merge the environment update with current environment
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

            # Generate image if enabled
            image_tool_result = None
            if self.enable_image_generation:
                # Actually generate the image using the shared image generator
                from agent.image_generation import get_shared_image_generator

                image_generator = get_shared_image_generator()

                def image_progress_callback(progress_data):
                    # Forward progress to the main progress callback
                    progress_callback(progress_data)

                image_tool_result = image_generator.generate_image_for_action(
                    image_description, llm, model, image_progress_callback
                )

            # Create action result with or without image generation
            image_result = None
            if image_tool_result:
                if isinstance(image_tool_result, ToolCallError):
                    raise Exception(
                        f"Image generation failed: {image_tool_result.error}"
                    )

                assert isinstance(
                    image_tool_result.content, ImageGenerationToolContent
                ), "Image generation must return ImageGenerationToolContent"

                image_result = image_tool_result.content
            else:
                image_result = ImageGenerationToolContent(
                    prompt=image_description,
                    image_path="",
                    image_url="/placeholder-image.png",
                    width=512,
                    height=512,
                    num_inference_steps=0,
                    guidance_scale=0.0,
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
