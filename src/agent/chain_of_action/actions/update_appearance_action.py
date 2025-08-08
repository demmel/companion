"""
UPDATE_APPEARANCE action implementation.
"""

from __future__ import annotations

import time
import logging

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.types import ImageGenerationToolContent, ToolCallError

from ..action_types import ActionType
from ..base_action import BaseAction
from ..context import ActionResult, ExecutionContext
from ..action_plan import ActionPlan

from agent.state import State
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UpdateAppearanceActionMetadata(BaseModel):
    """Metadata for the UPDATE_APPEARANCE action"""

    image_description: str
    old_appearance: str
    new_appearance: str
    image_result: ImageGenerationToolContent


class AppearanceUpdate(BaseModel):
    """Updated appearance description"""

    updated_appearance: str = Field(
        description="Updated appearance description in first-person that builds on current state. MUST start with 'I' and follow this exact order: 1) Pose/posture 2) Facial expression 3) Hair/body details 4) Clothing/accessories. PRIORITIZE emotional expression over clothing. Example: 'I'm leaning forward with a gentle smile, my hair falling softly around my shoulders, wearing a simple blue dress.' Always start with current appearance and modify only what was requested to change."
    )


def _build_image_description(
    appearance: str, environment: str, agent_name: str, llm: LLM, model: SupportedModel
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

    response = llm.generate_complete(model, prompt)
    return response.strip()


class UpdateAppearanceAction(BaseAction):
    """Update the agent's appearance and generate a new image"""

    action_type = ActionType.UPDATE_APPEARANCE

    def __init__(self, enable_image_generation: bool = True):
        self.enable_image_generation = enable_image_generation

    @classmethod
    def get_action_description(cls) -> str:
        return "Update my appearance (posture, expression, hair, clothing) and generate a new image"

    @classmethod
    def get_context_description(cls) -> str:
        return "Description of what specific aspects of my appearance should change and how"

    @classmethod
    def create_result_from_image_generation(
        cls,
        image_result: ImageGenerationToolContent,
        image_description: str,
        context_given: str,
        duration_ms: float,
        new_appearance: str,
        old_appearance: str = "",
        success: bool = True,
        result_summary: str = "Appearance updated with new image",
    ) -> "ActionResult[UpdateAppearanceActionMetadata]":
        """Factory method to create consistent ActionResult for UPDATE_APPEARANCE actions"""
        from agent.chain_of_action.base_action import ActionResult
        from agent.chain_of_action.action_types import ActionType

        return ActionResult(
            action=ActionType.UPDATE_APPEARANCE,
            result_summary=result_summary,
            context_given=context_given,
            duration_ms=duration_ms,
            success=success,
            metadata=UpdateAppearanceActionMetadata(
                image_description=image_description,
                old_appearance=old_appearance,
                new_appearance=new_appearance,
                image_result=image_result,
            ),
        )

    def execute(
        self,
        action_plan: ActionPlan,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        logger.debug("=== UPDATE_APPEARANCE ACTION ===")
        logger.debug(f"APPEARANCE CHANGE: {action_plan.context}")

        old_appearance = state.current_appearance

        # Use LLM to merge the appearance update with current appearance
        merge_prompt = f"""I need to update my appearance based on a specific change request. Please merge the requested changes with my current appearance.

Current appearance: "{old_appearance}"
Requested change: "{action_plan.context}"

Please provide an updated appearance description that:
1. Builds on my current appearance 
2. Incorporates the requested changes
3. Follows the format: pose/posture, facial expression, hair/body details, clothing/accessories
4. Uses first-person perspective starting with "I"
5. Prioritizes emotional expression over clothing details

The result should be a natural evolution of my current appearance with the requested modifications."""

        try:
            # Get updated appearance using structured LLM call
            appearance_update = direct_structured_llm_call(
                prompt=merge_prompt,
                response_model=AppearanceUpdate,
                model=model,
                llm=llm,
            )

            # Update state with new appearance
            state.current_appearance = appearance_update.updated_appearance

            # Generate image with updated appearance
            image_description = _build_image_description(
                state.current_appearance,
                state.current_environment,
                state.name,
                llm,
                model,
            )

            logger.debug(f"Generated image description: {image_description}")

            # Generate image if enabled
            image_result = None
            if self.enable_image_generation:
                # Actually generate the image using the shared image generator
                from agent.tools.image_generation import get_shared_image_generator

                image_generator = get_shared_image_generator()

                def image_progress_callback(progress_data):
                    # Forward progress to the main progress callback
                    progress_callback(progress_data)

                image_result = image_generator.generate_image_for_action(
                    image_description, llm, model, image_progress_callback
                )

            duration_ms = (time.time() - start_time) * 1000

            result_summary = (
                f"Appearance updated: {appearance_update.updated_appearance}"
            )

            # Create action result with or without image generation
            if image_result:
                if isinstance(image_result, ToolCallError):
                    raise Exception(f"Image generation failed: {image_result.error}")

                assert isinstance(
                    image_result.content, ImageGenerationToolContent
                ), "Image generation must return ImageGenerationToolContent"

                metadata = UpdateAppearanceActionMetadata(
                    image_description=image_description,
                    old_appearance=old_appearance,
                    new_appearance=appearance_update.updated_appearance,
                    image_result=image_result.content,
                )
            else:
                fake_image_content = ImageGenerationToolContent(
                    prompt=image_description,
                    image_path="",
                    image_url="/placeholder-image.png",
                    width=512,
                    height=512,
                    num_inference_steps=0,
                    guidance_scale=0.0,
                )

                metadata = UpdateAppearanceActionMetadata(
                    image_description=image_description,
                    old_appearance=old_appearance,
                    new_appearance=appearance_update.updated_appearance,
                    image_result=fake_image_content,
                )

            result = ActionResult(
                action=ActionType.UPDATE_APPEARANCE,
                result_summary=result_summary,
                context_given=action_plan.context,
                duration_ms=duration_ms,
                success=True,
                metadata=metadata,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to update appearance: {e}")

            return ActionResult(
                action=ActionType.UPDATE_APPEARANCE,
                result_summary=f"Failed to update appearance: {str(e)}",
                context_given=action_plan.context,
                duration_ms=duration_ms,
                success=False,
                metadata=None,  # No metadata on error
            )
