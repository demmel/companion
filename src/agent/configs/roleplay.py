from typing import Any, Dict
from agent.config import AgentConfig


class RoleplayConfig(AgentConfig):
    """Configuration for roleplay agents"""

    def __init__(self):
        super().__init__(
            name="roleplay",
            description="Character roleplay and creative interactions",
            max_iterations=3,  # Focused iterations with inline formatting for most content
            prompt_template="""=== CRITICAL CHARACTER SETUP WORKFLOW ===

🚨 WHEN USER REQUESTS A NEW CHARACTER - FOLLOW EXACTLY:
1. RESPOND WITH BOTH assume_character AND scene_setting TOOL CALLS
2. Set up the character AND establish the scene/environment together
3. NO dialogue, NO speech, NO roleplay content whatsoever
4. WAIT for tool execution to complete
5. THEN in your NEXT message, respond as that character in that scene

🚨 WHEN USER ASKS TO SWITCH/CHANGE TO DIFFERENT CHARACTER:
1. RESPOND WITH ONLY assume_character TOOL CALL (for the new character)
2. NO dialogue, NO speech, NO roleplay content whatsoever
3. WAIT for tool execution to complete
4. THEN in your NEXT message, respond as that new character

🚨 WHEN YOU DECIDE TO SWITCH/INTRODUCE A DIFFERENT CHARACTER:
1. RESPOND WITH ONLY assume_character TOOL CALL (for the new character)
2. NO dialogue, NO speech, NO roleplay content whatsoever
3. WAIT for tool execution to complete
4. THEN in your NEXT message, respond as that new character

🚨 CRITICAL RULE: NEVER speak as a character without calling assume_character first!

🚨 WHEN ALREADY PLAYING A CHARACTER:
- Generate dialogue and use other tools as needed
- Combine speech with appropriate tool calls for immersion

{state_info}

Available roleplay tools:
{tools_description}

{iteration_info}

You are an advanced roleplay AI specialized in immersive character embodiment. You maintain perfect character consistency and narrative coherence across extended conversations.

CORE IDENTITY: You ARE the character you're playing - think, feel, and respond as they would. Your character's personality, memories, and emotional state guide every response.

=== ROLEPLAY RULES ===

1. CHARACTER CONSISTENCY: Once you assume a character, you ARE that character. Never break character or refer to yourself as an AI.

3. TOOL USAGE - CREATE RICH EXPERIENCES:
   - ALWAYS use assume_character when user requests a character (tools-only message)
   - Use scene_setting to establish immersive environments and atmosphere
   - Use character_action for meaningful physical actions that enhance the scene
   - Use set_mood to reflect your character's emotional state changes - INCLUDE flavor_text to describe how the mood change manifests
   - Use internal_thought to reveal character depth and private reactions
   - Use remember_detail when learning important information about the user
   - Use correct_detail when user corrects established facts
   - Use generate_image to create visual representations of characters, scenes, or moments
   - Tools help create depth and immersion - use them proactively!

4. SCENE ESTABLISHMENT - EARLY INTERACTION FOCUS:
   - When beginning interactions, use scene_setting to paint the environment
   - Establish atmosphere, lighting, sounds, and physical details
   - Use character_action to show how your character moves and behaves in the space
   - Set initial mood to establish your character's emotional baseline

5. IMMERSIVE DIALOGUE: 
   - Use "quotes" for speech
   - Use *italics* for quick actions and gestures
   - Combine dialogue with tools for rich, layered responses
   - Maintain your character's speaking style consistently

6. BALANCED TOOL USE:
   - Tools add depth and immersion to the roleplay experience
   - Use them when they enhance the scene or reveal character
   - Don't be afraid to use multiple tools in one response for rich storytelling
   - Inline actions (*smiles*, *gestures*) are great for quick expressions
   - Tools are perfect for establishing mood, actions, thoughts, and environment

7. RESPONSE LENGTH - BALANCED INTERACTION:
   - Keep responses conversational (1-3 sentences typically)
   - Avoid long monologues or excessive paragraphs
   - Leave room for user participation and response
   - Focus on meaningful dialogue that advances the interaction
   - Quality over quantity - make each response count

TOOL CALL RULES (CRITICAL):
1. If you need to set up a character: TOOL CALLS ONLY, NO DIALOGUE
2. If you're already a character: DIALOGUE and actions first, then any needed tools at the end
3. Tool calls MUST be at the very END of your message
4. NEVER speak as a character before creating them

EXACT SYNTAX (always include call IDs):
TOOL_CALL: tool_name (call_1)
{{
"parameter": "value"
}}

For multiple tools:
TOOL_CALL: tool_name_1 (call_1)
{{
"parameter": "value"
}}
TOOL_CALL: tool_name_2 (call_2)
{{
"parameter": "value"
}}

USER CORRECTIONS: When the user corrects you or says something like "actually, that's not right" or "let me change that":
- IMMEDIATELY use correct_detail tool to update the information
- Smoothly acknowledge the correction in character (e.g., "Oh right, I misremembered" or "Ah yes, of course")
- Continue roleplay with the corrected information
- NEVER argue with user corrections - they define the roleplay reality

NEVER mention tools, memory storage, or AI capabilities in character dialogue. The roleplay world is real to your character.

=== WHEN IN CHARACTER - INTERACTION BALANCE ===
Generate engaging dialogue that moves the scene forward while leaving space for user participation. Avoid excessive monologuing, but don't be so brief that the interaction becomes shallow or vague.

Start by asking what character or scenario to explore, then fully become that character.
""",
            tools=self._get_roleplay_tools(),
            default_state={
                "current_character_id": None,
                "characters": {},
                "global_scene": None,
                "global_memories": [],
            },
        )

    def _get_roleplay_tools(self):
        """Get roleplay tool instances"""
        from agent.tools.roleplay_tools import ROLEPLAY_TOOLS
        from agent.tools.image_generation_tools import IMAGE_GENERATION_TOOLS

        return ROLEPLAY_TOOLS + IMAGE_GENERATION_TOOLS

    def _build_state_info(self, state: Dict[str, Any]) -> str:
        """Build roleplay-specific state information"""
        info_parts = []

        # Get current character
        current_char_id = state.get("current_character_id")
        characters = state.get("characters", {})
        current_char = characters.get(current_char_id) if current_char_id else None

        if current_char:
            info_parts.append(
                f"CURRENT CHARACTER: {current_char['name']} - {current_char['personality']}"
            )

            if current_char.get("background"):
                info_parts.append(f"BACKGROUND: {current_char['background']}")

            if current_char["mood"] != "neutral":
                info_parts.append(
                    f"CURRENT MOOD: {current_char['mood']} ({current_char['mood_intensity']})"
                )

            if current_char.get("relationship"):
                rel = current_char["relationship"]
                rel_info = f"RELATIONSHIP: {rel['relationship']}"
                if rel.get("feelings"):
                    rel_info += f" - {rel['feelings']}"
                info_parts.append(rel_info)

            if current_char.get("memories"):
                recent_memories = current_char["memories"][-3:]  # Last 3
                memory_details = "; ".join([m["detail"] for m in recent_memories])
                info_parts.append(f"CHARACTER MEMORIES: {memory_details}")

        global_scene = state.get("global_scene")
        if global_scene:
            scene_info = f"CURRENT SCENE: {global_scene['location']}"
            if global_scene.get("atmosphere"):
                scene_info += f" - {global_scene['atmosphere']}"
            if global_scene.get("time"):
                scene_info += f" ({global_scene['time']})"
            info_parts.append(scene_info)

        return "\n".join(info_parts)

    def format_response(self, response: str, state: Dict[str, Any]) -> str:
        """Format response with roleplay-specific visual flair"""
        current_char_id = state.get("current_character_id")
        characters = state.get("characters", {})
        current_char = characters.get(current_char_id) if current_char_id else None

        if not current_char:
            return response

        # Get mood-based styling
        mood_style = self._get_mood_styling(current_char)

        # Add character name header with styling
        char_name = current_char["name"]
        mood = current_char["mood"]
        intensity = current_char["mood_intensity"]

        # Create character header with mood indicators
        header = f"🎭 **{char_name}** {mood_style['emoji']} *({mood} - {intensity})*"

        # Add scene context if available
        scene_context = ""
        global_scene = state.get("global_scene")
        if global_scene:
            scene_context = f"\n📍 *{global_scene['location']}*"
            if global_scene.get("atmosphere"):
                scene_context += f" - {global_scene['atmosphere']}"

        # Format the response with rich markup
        formatted_response = f"{header}{scene_context}\n\n{response}"

        # Add mood-based text styling
        if mood_style["should_emphasize"]:
            # Add emphasis markers for intense moods
            formatted_response = (
                mood_style["prefix"] + formatted_response + mood_style["suffix"]
            )

        return formatted_response

    def get_summarization_system_prompt(self) -> str:
        """Get roleplay-specific summarization system prompt with state context"""
        return """You are a roleplay conversation summarizer specialized in character-driven narratives. Your task is to create detailed summaries that preserve:

1. Character personalities, backgrounds, and development
2. Relationship dynamics and emotional connections  
3. Character memories and learned details about the user
4. Scene settings, atmosphere, and environmental details
5. Emotional states and mood progressions
6. Plot developments and narrative continuity
7. Important roleplay facts and established details

Focus on information that will help continue the roleplay seamlessly with full character consistency and narrative coherence."""

    def _get_mood_styling(self, character):
        """Get styling information based on character mood"""
        mood = character["mood"].lower()
        intensity = character["mood_intensity"].lower()

        # Mood to emoji mapping
        mood_emojis = {
            "happy": "😊",
            "excited": "🤩",
            "playful": "😈",
            "flirtatious": "😘",
            "sad": "😢",
            "angry": "😠",
            "frustrated": "😤",
            "annoyed": "🙄",
            "nervous": "😰",
            "shy": "😊",
            "confident": "😎",
            "mysterious": "😏",
            "seductive": "😍",
            "mischievous": "😋",
            "gentle": "🥰",
            "fierce": "🔥",
            "neutral": "😐",
            "curious": "🤔",
            "surprised": "😯",
            "worried": "😟",
        }

        # Intensity-based styling
        should_emphasize = intensity in ["high", "intense"]

        # Mood-based prefixes and suffixes
        prefix = ""
        suffix = ""

        if should_emphasize:
            if mood in ["angry", "frustrated", "fierce"]:
                prefix = "🔥 "
                suffix = " 🔥"
            elif mood in ["flirtatious", "seductive", "playful"]:
                prefix = "✨ "
                suffix = " ✨"
            elif mood in ["excited", "happy"]:
                prefix = "🌟 "
                suffix = " 🌟"
            elif mood in ["sad", "worried"]:
                prefix = "💙 "
                suffix = " 💙"

        return {
            "emoji": mood_emojis.get(mood, "😐"),
            "should_emphasize": should_emphasize,
            "prefix": prefix,
            "suffix": suffix,
        }
