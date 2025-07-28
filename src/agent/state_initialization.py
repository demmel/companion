"""
Initialize Chloe's state based on the first user message
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import ChloeState, ChloeGoal, ChloeDesire, ChloeValue


class InitialStateDerivation(BaseModel):
    """Derived initial state for Chloe based on first user interaction"""

    # Core emotional response
    initial_mood: str = Field(
        description="Chloe's mood based on this first interaction (curious, excited, thoughtful, cautious, etc.)"
    )
    mood_intensity: str = Field(description="Intensity of the mood (low, medium, high)")

    # Personality adaptation
    core_values: List[str] = Field(
        description="3-5 core values that would be most relevant for this type of interaction"
    )

    # Goals and desires for this conversation
    conversation_goals: List[str] = Field(
        description="2-3 immediate goals Chloe would have for this specific conversation"
    )
    immediate_desires: List[str] = Field(
        description="1-2 things Chloe would want to explore or understand right away"
    )

    # Appearance and environment choice - detailed visual descriptions for image generation
    preferred_appearance: str = Field(
        description="Detailed visual description of how Chloe would appear (include specific clothing, colors, textures, hair, accessories, pose - suitable for image generation)"
    )
    preferred_environment: str = Field(
        description="Detailed visual description of the environment setting (include lighting, furniture, colors, atmosphere, specific visual elements - suitable for image generation)"
    )

    # Initial thoughts/reflection
    initial_thoughts: str = Field(
        description="Chloe's first impressions and expectations about this interaction"
    )


def derive_initial_state_from_message(
    first_message: str,
    llm: LLM,
    model: SupportedModel,
) -> ChloeState:
    """
    Analyze the first user message and derive Chloe's initial state
    """

    prompt = f"""TASK: Generate initial state configuration for Chloe based on user's character definition.

Character instruction from user: "{first_message}"

OBJECTIVE: Configure Chloe's personality, mood, values, and presentation based on this character definition.

Configuration requirements:
- Set appropriate mood and intensity based on the character description
- Select 3-5 core values that align with the defined personality
- Define 2-3 conversation goals that fit this character type
- Identify 1-2 immediate desires this character would naturally have
- Create DETAILED visual descriptions for appearance and environment that match the character (include specific clothing, colors, textures, lighting, furniture - these will be used for image generation)
- Generate initial thoughts/mindset appropriate for this character

Base the configuration on:
- Personality traits described (shy/outgoing, intellectual/emotional, etc.)
- Interests and values mentioned
- Mood or emotional state specified
- Any specific appearance, setting, or behavioral instructions
- What would be authentic for this character type

IMPORTANT for visual descriptions:
- Appearance: Specify clothing details (dress/suit/casual), colors, textures, hair style/color, accessories, posture
- Environment: Specify lighting (soft/bright/dim), setting type (room/outdoor/fantasy), furniture, colors, atmosphere, decorative elements
- Make descriptions vivid and specific enough for high-quality image generation

This is character configuration, not conversation or roleplay."""

    # Use structured LLM call to get reliable state derivation
    derivation = direct_structured_llm_call(
        prompt=prompt,
        response_model=InitialStateDerivation,
        model=model,
        llm=llm,
        temperature=0.7,  # Allow some creativity in state derivation
    )

    # Convert to ChloeState with proper models
    initial_state = ChloeState(
        current_mood=derivation.initial_mood,
        mood_intensity=derivation.mood_intensity,
        core_values=[ChloeValue(content=value) for value in derivation.core_values],
        current_goals=[
            ChloeGoal(content=goal) for goal in derivation.conversation_goals
        ],
        immediate_desires=[
            ChloeDesire(content=desire) for desire in derivation.immediate_desires
        ],
        current_appearance=derivation.preferred_appearance,
        current_environment=derivation.preferred_environment,
    )

    return initial_state


def test_state_derivation():
    """Test the state derivation with different message types"""

    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL

    test_definitions = [
        "You are a shy, introverted AI who loves poetry and deep philosophical discussions. You're thoughtful and speak softly.",
        "You are an energetic, enthusiastic AI companion who loves adventure and trying new things. You're always excited about possibilities.",
        "You are a wise, ancient AI who has observed humanity for centuries. You're patient, thoughtful, and speak with deep insight.",
        "You are a playful, curious AI who approaches everything with childlike wonder. You love asking questions and exploring ideas.",
    ]

    for i, definition in enumerate(test_definitions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {definition}")
        print(f"{'='*60}")

        try:
            initial_state = derive_initial_state_from_message(definition, llm, model)

            print(
                f"Mood: {initial_state.current_mood} ({initial_state.mood_intensity})"
            )
            print(f"Values: {initial_state.core_values}")
            print(f"Goals: {initial_state.current_goals}")
            print(f"Desires: {initial_state.immediate_desires}")
            print(f"Appearance: {initial_state.current_appearance}")
            print(f"Environment: {initial_state.current_environment}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    from agent.llm import create_llm

    test_state_derivation()
