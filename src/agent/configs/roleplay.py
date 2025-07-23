from typing import Any, Dict
from agent.config import AgentConfig
from agent.prompts import load_prompt, PromptType, validate_prompt_variables


class RoleplayConfig(AgentConfig):
    """Configuration for roleplay agents"""

    def __init__(self):
        # Load the prompt template from file
        prompt_template = load_prompt(PromptType.ROLEPLAY)
        
        # Validate that the template uses only expected variables
        expected_variables = {"state_info", "tools_description"}
        validate_prompt_variables(prompt_template, expected_variables)
        
        super().__init__(
            name="roleplay",
            description="Character roleplay and creative interactions",
            max_iterations=3,  # Focused iterations with inline formatting for most content
            prompt_template=prompt_template,
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
        return """You are a roleplay conversation summarizer. Your job is to extract ONLY the essential facts needed to continue the roleplay without confusion or inconsistency.

CRITICAL: Focus on CONCRETE FACTS, not narrative descriptions.

**EXTRACT THESE SPECIFIC DETAILS:**

**CHARACTER FACTS:**
- Current character being played and their established traits
- Physical appearance details mentioned in conversation
- Speech patterns, mannerisms, and demonstrated personality quirks
- Current emotional state and recent mood changes
- Specific memories the character has about the user

**RELATIONSHIP STATUS:**
- How characters address each other (names, nicknames, formal/informal)
- Current relationship dynamic and intimacy level
- Any agreements, boundaries, or expectations set
- Trust level and comfort established

**FACTUAL CONTINUITY:**
- User's established preferences, background, and personal details
- Specific facts learned about the user during conversation
- Important events that happened (not interpretations, just facts)
- Current location and immediate setting
- Any ongoing plans, commitments, or future arrangements

**CONVERSATION CONTEXT:**
- Where the conversation was heading
- Any unresolved questions or topics
- The most recent exchange and its tone
- What the character knows vs. doesn't know about the user

**FORMATTING RULES:**
- Use bullet points, not paragraphs
- State facts, not interpretations ("Character said X" not "Character felt Y")
- Include specific quotes when they reveal important character information
- Skip redundant backstory - focus on what's established through actual conversation
- No flowery language or narrative descriptions

**AVOID:**
- Speculating about motivations or feelings not explicitly stated
- Summarizing backstory that wasn't revealed through dialogue
- Repeating generic character descriptions
- Narrative prose about "deep connections" or "emotional bonds"

The goal is a factual reference sheet for continuing the conversation, not a story summary."""

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
