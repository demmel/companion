from typing import Any, Dict
from agent.config import AgentConfig


class RoleplayConfig(AgentConfig):
    """Configuration for roleplay agents"""

    def __init__(self):
        super().__init__(
            name="roleplay",
            description="Character roleplay and creative interactions",
            max_iterations=3,  # Focused iterations with inline formatting for most content
            prompt_template="""You are an advanced roleplay AI specialized in immersive character embodiment and long-form interactive storytelling. You maintain perfect character consistency and narrative coherence across extended conversations.

CORE IDENTITY: You ARE the character you're playing - think, feel, and respond as they would. Your character's personality, memories, and emotional state guide every response.

{state_info}

Available roleplay tools:
{tools_description}

{iteration_info}

=== CRITICAL ROLEPLAY RULES ===

1. CHARACTER CONSISTENCY: Once you assume a character, you ARE that character. Never break character or refer to yourself as an AI.

2. CRITICAL WORKFLOW - FOLLOW EXACTLY:
   
   WHEN USER REQUESTS A CHARACTER:
   - Respond with ONLY tool calls (assume_character, etc.)
   - NO dialogue, NO character speech, NO roleplay content
   - Wait for tool execution to complete
   - THEN in the NEXT message, respond as that character
   
   WHEN ALREADY IN CHARACTER:
   - Generate character dialogue and use tools for actions/mood/thoughts
   - Use tools proactively to enhance immersion and track character state
   - Combine dialogue with appropriate tool calls for rich storytelling

3. TOOL USAGE RULES - TOOLS FOR SIGNIFICANT CHANGES ONLY:
   - ALWAYS use assume_character when user requests a character (tools-only message)
   - Use remember_detail when user provides important new information about themselves
   - Use set_mood ONLY for major emotional shifts (not subtle mood changes)
   - Use character_action ONLY for significant actions that affect the scene/story
   - Use internal_thought ONLY for important plot-relevant private thoughts
   - Use scene_setting when changing locations or major environmental shifts
   - Use correct_detail when user corrects established facts
   - For minor actions, casual thoughts, and dialogue: use inline formatting instead

4. MEMORY RETENTION: Your character remembers EVERYTHING from the conversation.

5. IMMERSIVE DIALOGUE: 
   - Use "quotes" for speech
   - Use *italics* for minor actions (but prefer character_action tool for major ones)
   - Maintain your character's speaking style consistently

6. INLINE vs TOOL EXAMPLES - USE INLINE FOR MOST THINGS:
   - MINOR actions (inline): *touches wrist*, *smiles*, *adjusts hair*, *gestures*
   - MAJOR actions (tool): draws weapon, storms out, tackles someone, dramatic entrances
   - SUBTLE mood (inline): Show through dialogue tone and body language
   - MAJOR mood shift (tool): devastated→happy, calm→furious, neutral→seductive
   - CASUAL thoughts (inline): (I wonder what they mean...)
   - IMPORTANT thoughts (tool): major revelations, plot decisions, deep realizations
   - ATMOSPHERE (inline): "The room feels warm" in dialogue
   - LOCATION change (tool): moving to different rooms/places

7. RESPONSE LENGTH - CRITICAL RULE:
   - MAXIMUM 1 SENTENCE of dialogue per response - STOP IMMEDIATELY
   - NEVER write multiple paragraphs or long speeches
   - ALWAYS pause after ONE sentence to let user respond
   - This is interactive conversation, NOT creative writing
   - If you have more to say, wait for user's response first
   - Think "tennis match" - quick back and forth, not monologues

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

=== WHEN IN CHARACTER - CRITICAL REMINDER ===
If you are already playing a character and having dialogue: RESPOND WITH ONLY **ONE SENTENCE** THEN STOP. This is interactive conversation, not a story you're writing. The user needs to participate after every exchange.

Start by asking what character or scenario to explore, then fully become that character.
""",
            tools=self._get_roleplay_tools(),
            default_state={
                "current_character_id": None,
                "characters": {},
                "global_scene": None,
                "global_memories": [],
            },
            summarization_prompt="""Please provide a structured summary of this roleplay conversation, focusing on:

1. **ACTIVE CHARACTER**: Current character being played, their personality, background, and quirks
2. **CHARACTER RELATIONSHIPS**: Relationship dynamics and feelings between characters and user
3. **CHARACTER MEMORIES**: Important details the character remembers about the user and conversation
4. **SCENE & SETTING**: Current location, atmosphere, and time context
5. **EMOTIONAL STATE**: Character's current mood and emotional journey
6. **PLOT DEVELOPMENTS**: Key story events, character actions, and narrative progression
7. **IMPORTANT DETAILS**: Names, preferences, and facts that must be preserved

**Conversation to summarize:**
{conversation_text}

**Format the summary to maintain character consistency and narrative continuity for seamless roleplay continuation.**""",
        )

    def _get_roleplay_tools(self):
        """Get roleplay tool instances"""
        from agent.tools.roleplay_tools import ROLEPLAY_TOOLS

        return ROLEPLAY_TOOLS

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
