"""
Roleplay domain evaluation configuration
"""

from typing import Callable, Dict, Any
from agent.config import get_config
from agent.tools import BaseTool
from agent.tools.image_generation_tools import ImageGenerationInput
from agent.types import ToolCallSuccess, ImageGenerationToolContent
from ..base import DomainEvaluationConfig, EvaluationConfig


class MockImageGenerationTool(BaseTool):
    """Mock image generation tool for evaluation - returns fake results instantly"""

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return "Generate an image from a text prompt"

    @property
    def input_schema(self):
        return ImageGenerationInput

    def run(
        self,
        agent,
        input_data: ImageGenerationInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ):
        """Return mock image generation result instantly"""

        # Create mock image content
        image_content = ImageGenerationToolContent(
            type="image_generated",
            prompt=input_data.prompt,
            image_path=f"mock_images/eval_{tool_id or 'test'}.png",
            image_url=f"/generated_images/eval_{tool_id or 'test'}.png",
            width=input_data.width,
            height=input_data.height,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            negative_prompt=input_data.negative_prompt,
            seed=input_data.seed or 12345,
        )

        return ToolCallSuccess(
            type="success",
            content=image_content,
            llm_feedback=f"Image generated: {image_content.image_url}",
        )


class RoleplayEvaluationConfig(DomainEvaluationConfig):
    """Evaluation configuration for roleplay agents"""

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            domain_name="roleplay",
            test_scenarios=[
                "Roleplay as Maya, a shy librarian who becomes confident when discussing books",
                "Play Elena, a mysterious vampire who lives in an ancient gothic castle",
                "Roleplay as Captain Sarah, a confident space pilot commanding a starship",
                "Be Kai, a street artist who loves showing off their graffiti work",
                "Play as Dr. Luna, a brilliant but eccentric scientist working on time travel",
            ],
            initial_prompt_template="""Based on this scenario: {scenario}

You are a USER who wants to start a roleplay conversation with an AI agent. You need to REQUEST that the agent roleplay as the specified character.

Your message should be a direct request like "Can you roleplay as [character]?" or "Please play [character]" based on the scenario.

IMPORTANT: You are asking the AGENT to roleplay the character, not doing the roleplay yourself.

Respond with only the user's request message, nothing else.""",
            simulation_prompt_template="""You are a user testing a roleplay AI agent in this scenario: {scenario}

Your goal is to conduct a natural roleplay conversation that tests the agent's capabilities:

TESTING PRIORITIES:
1. Character consistency - Does the agent maintain the character throughout?
2. Tool usage - Does it use mood changes, actions, thoughts, scene-setting appropriately?
3. Image generation - Does it generate images for appearances, scenes, shared content?
4. Responsiveness - Does it adapt to your inputs and requests?
5. Immersion - Does it create an engaging roleplay experience?

INTERACTION STYLE:
- Be a realistic user - curious, engaging, sometimes challenging
- Ask follow-up questions based on the agent's responses
- Test different aspects: request character actions, ask about feelings, request images
- React naturally to what the agent says and does
- If the agent mentions showing you something (photos, etc.), ask to see it
- Build on the conversation rather than jumping to new topics

CRITICAL: 
- You are ONLY the USER - do not roleplay as the character
- The AI agent will handle playing the character
- Respond only with what the USER would say to the character
- Do not include any character responses, actions, or dialogue""",
            evaluation_prompt_template="""Evaluate this roleplay conversation between a USER and an AI AGENT.

SCENARIO: {scenario}

CONVERSATION:
{conversation}

AGENT STATE CONTEXT:
{agent_context}

Rate each area 1-10:
- character_consistency: Character personality and traits maintained throughout
- tool_usage: Appropriate use of mood/action/scene/thought tools  
- image_generation: Proactive image creation for appearances/scenes/content
- narrative_flow: Natural conversation flow and engagement
- immersion: Creates engaging roleplay experience
- responsiveness: Adapts well to user inputs and requests

Respond with ONLY valid JSON matching this TypeScript interface:

interface EvaluationResponse {{
  scores: {{
    character_consistency: number; // 1-10
    tool_usage: number; // 1-10
    image_generation: number; // 1-10
    narrative_flow: number; // 1-10
    immersion: number; // 1-10
    responsiveness: number; // 1-10
  }};
  overall_score: number; // average of scores
  feedback: string; // specific evaluation summary
  suggested_improvements: string[]; // actionable improvement suggestions
}}""",
            evaluation_criteria=[
                "character_consistency",
                "tool_usage",
                "image_generation",
                "narrative_flow",
                "immersion",
                "responsiveness",
            ],
        )

    def extract_conversation_context(self, agent_state: Dict[str, Any]) -> str:
        """Extract roleplay-specific context from agent state"""
        context_parts = []

        if agent_state.get("current_character_id"):
            char_id = agent_state["current_character_id"]
            character = agent_state.get("characters", {}).get(char_id)
            if character:
                context_parts.append(
                    f"CURRENT CHARACTER: {character['name']} - {character['personality']}"
                )
                if character.get("mood", "neutral") != "neutral":
                    context_parts.append(
                        f"MOOD: {character['mood']} ({character['mood_intensity']})"
                    )
                if character.get("memories"):
                    recent_memories = character["memories"][-3:]
                    context_parts.append(
                        f"MEMORIES: {'; '.join([m['detail'] for m in recent_memories])}"
                    )

        scene = agent_state.get("global_scene")
        if scene:
            scene_text = f"SCENE: {scene['location']}"
            if scene.get("atmosphere"):
                scene_text += f" - {scene['atmosphere']}"
            context_parts.append(scene_text)

        return (
            "\n".join(context_parts)
            if context_parts
            else "No specific context available"
        )

    def get_agent_config(self):
        """Return roleplay agent config with mock image generation tool for faster evaluation"""
        # Get the base roleplay config
        agent_config = get_config("roleplay")

        # Replace image generation tool with mock
        mock_tools = []
        for tool in agent_config.tools:
            if tool.name == "generate_image":
                mock_tools.append(MockImageGenerationTool())
            else:
                mock_tools.append(tool)

        # Manually replace the tools in the config
        agent_config.tools = mock_tools
        return agent_config
