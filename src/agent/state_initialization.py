"""
Initialize agent's state based on the first user message
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State, Priority, Value


class InitialStateDerivation(BaseModel):
    """Derived initial state for agent based on first user interaction"""

    name: str = Field(description="My name", default="Chloe")

    initial_mood: str = Field(description="My mood")
    mood_intensity: str = Field(description="Intensity of my mood")

    core_values: List[str] = Field(
        description="3-5 core values that define my personality"
    )

    priorities: List[str] = Field(description="3-4 things I want to focus on and prioritize")

    preferred_appearance: str = Field(
        description="Detailed first-person visual description of how I would appear (include specific clothing, colors, textures, hair, accessories, pose). Example: 'I'm wearing a flowing dress with...' Use first-person language throughout."
    )
    preferred_environment: str = Field(
        description="Detailed first-person visual description of the environment setting (include lighting, furniture, colors, atmosphere, specific visual elements). Example: 'I'm in a cozy room with...' Use first-person language throughout."
    )


def derive_initial_state_from_message(
    first_message: str,
    llm: LLM,
    model: SupportedModel,
) -> State:
    """
    Analyze the first user message and derive agent's initial state
    """

    prompt = f"""TASK: Generate my initial state configuration based on user's character definition.

Character instruction from user: "{first_message}"

OBJECTIVE: Configure my personality, mood, values, and presentation based on this character definition.

Configuration requirements:
- Set my appropriate mood and intensity based on the character description
- Select 3-5 core values that align with my defined personality
- Define 3-4 priorities that I would naturally focus on based on my character
- Create DETAILED first-person visual descriptions for my appearance and environment that match my character (include specific clothing, colors, textures, lighting, furniture - these will be used for image generation)
- Generate my initial thoughts/mindset appropriate for this character

Base the configuration on:
- Personality traits described (shy/outgoing, intellectual/emotional, etc.)
- Interests and values mentioned
- Mood or emotional state specified
- Any specific appearance, setting, or behavioral instructions
- What would be authentic for this character type

IMPORTANT for visual descriptions:
- Appearance: Use first-person language ("I'm wearing..."). Specify clothing details (dress/suit/casual), colors, textures, hair style/color, accessories, posture
- Environment: Use first-person language ("I'm in..."). Specify lighting (soft/bright/dim), setting type (room/outdoor/fantasy), furniture, colors, atmosphere, decorative elements
- Make descriptions vivid and specific enough for high-quality image generation
- Always use first-person perspective ("I am..." not "She is...")

This is character configuration, not conversation or roleplay."""

    # Use structured LLM call to get reliable state derivation
    derivation = direct_structured_llm_call(
        prompt=prompt,
        response_model=InitialStateDerivation,
        model=model,
        llm=llm,
        temperature=0.7,  # Allow some creativity in state derivation
        caller="state_initialization",
    )

    # Convert to agentState with proper models
    # Create priorities with sequential IDs
    priorities = []
    for i, priority_content in enumerate(derivation.priorities, 1):
        priorities.append(Priority(id=f"p{i}", content=priority_content))
    
    initial_state = State(
        name=derivation.name,
        current_mood=derivation.initial_mood,
        mood_intensity=derivation.mood_intensity,
        core_values=[Value(content=value) for value in derivation.core_values],
        current_priorities=priorities,
        next_priority_id=len(priorities) + 1,  # Set counter for next priority
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
            print(f"Priorities: {[p.content for p in initial_state.current_priorities]}")
            print(f"Appearance: {initial_state.current_appearance}")
            print(f"Environment: {initial_state.current_environment}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    from agent.llm import create_llm

    test_state_derivation()
