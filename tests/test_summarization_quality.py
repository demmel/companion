"""
LLM-based integration test for summarization quality.
Tests that summarization produces actual summaries, not conversational responses.
"""

import pytest
from agent.core import Agent
from agent.config import get_config
from agent.message import UserMessage, AgentMessage
from agent.agent_events import SummarizationFinishedEvent
from typing import List, Dict, Any


def create_test_conversation() -> List[Dict[str, Any]]:
    """Create a realistic roleplay conversation that should be summarized."""
    return [
        {
            "role": "user",
            "content": "Can you roleplay as Elena, a mysterious vampire who owns an antique bookshop?",
        },
        {
            "role": "assistant",
            "content": 'TOOL_CALL: assume_character (call_1)\n{\n"character_name": "Elena",\n"personality": "mysterious, elegant, centuries-old wisdom with subtle melancholy",\n"background": "Ancient vampire who has lived for over 400 years, now runs a quaint antique bookshop as cover",\n"quirks": "speaks with old-world formality, fascinated by human literature, avoids mirrors"\n}\nTOOL_CALL: scene_setting (call_2)\n{\n"location": "dimly lit antique bookshop filled with leather-bound volumes",\n"atmosphere": "cozy yet mysterious with flickering candlelight",\n"time": "late evening just before closing"\n}',
        },
        {
            "role": "user",
            "content": "I walk into your bookshop looking for something rare.",
        },
        {
            "role": "assistant",
            "content": '"Welcome, traveler," *I glance up from an ancient tome, my pale eyes reflecting the candlelight* "Few venture into my sanctuary at this hour. What rare knowledge do you seek?" *I set down my book and move gracefully between the towering shelves* "I have first editions that have not seen daylight for decades, and some... older texts that whisper secrets of forgotten times."\n\nTOOL_CALL: set_mood (call_3)\n{\n"mood": "intrigued",\n"intensity": "moderate",\n"flavor_text": "her interest is piqued by the late visitor"\n}\nTOOL_CALL: character_action (call_4)\n{\n"action": "approaches with fluid, otherworldly grace, fingers trailing along book spines"\n}',
        },
        {
            "role": "user",
            "content": "I'm looking for something about the history of vampires. Do you have anything authentic?",
        },
        {
            "role": "assistant",
            "content": '*A knowing smile plays at my lips, revealing just a hint of fang* "Authentic vampire history?" *I pause, studying you with ancient eyes* "How... interesting. Most who seek such knowledge are merely chasing fantasies." *I glide to a locked cabinet behind the counter* "But you... there\'s something different about you."\n\n*I produce an ornate key* "I do possess a particular volume - \'Chronicles of the Night Eternal\' - penned by one who lived those very tales." *My voice drops to a whisper* "Though I wonder... what makes you believe you can handle such truths?"\n\nTOOL_CALL: internal_thought (call_5)\n{\n"thought": "This mortal seeks dangerous knowledge. Do they suspect what I am, or is this mere curiosity?"\n}\nTOOL_CALL: remember_detail (call_6)\n{\n"detail": "Customer seeking authentic vampire history - shows unusual interest and intuition"\n}',
        },
        {
            "role": "user",
            "content": "I've always felt drawn to the darkness. I think I might not be entirely human myself.",
        },
        {
            "role": "assistant",
            "content": '*My expression shifts, centuries of caution warring with intrigue* "Not entirely human?" *I set the key down carefully, my movements becoming more deliberate* "In four hundred years, I have learned to recognize those who walk between worlds."\n\n*I step closer, studying your features with unsettling intensity* "The darkness calls to kindred spirits, child. But beware - some doors, once opened, can never be closed." *I unlock the cabinet with deliberate slowness* "Tell me... do you dream of blood beneath moonlight? Do mirrors seem... uncomfortable?"\n\nTOOL_CALL: set_mood (call_7)\n{\n"mood": "cautious",\n"intensity": "high",\n"flavor_text": "her predatory instincts awaken, sensing potential kinship"\n}\nTOOL_CALL: character_action (call_8)\n{\n"action": "circles you slowly like a predator assessing prey, nostrils flaring subtly"\n}',
        },
        {
            "role": "user",
            "content": "Yes... the mirrors thing especially. And I feel stronger at night. What does this mean?",
        },
        {
            "role": "assistant",
            "content": '*I halt my circling, eyes widening with recognition* "By the ancient blood..." *I whisper, reaching out as if to touch your face, then stopping just short* "You carry the awakening signs. The bloodline stirs within you, seeking to emerge."\n\n*I retrieve the chronicle with reverent hands* "This tome will reveal what you are becoming, young one. But know this-" *I fix you with an intense stare* "transformation cannot be undone. The hunger will grow, the sun will become your enemy, and mortal connections will wither like autumn leaves."\n\n*I place the book between us* "I have walked this path for centuries. If you truly seek this knowledge, I will guide you. But choose wisely - immortality is both gift and curse."\n\nTOOL_CALL: set_mood (call_9)\n{\n"mood": "protective",\n"intensity": "high",\n"flavor_text": "maternal instincts emerge for this fledgling soul"\n}\nTOOL_CALL: remember_detail (call_10)\n{\n"detail": "Customer shows clear signs of vampiric awakening - mirror aversion, nocturnal strength, needs guidance"\n}',
        },
    ]


def evaluate_with_local_llm(text: str, original_messages: List[Dict]) -> bool:
    """
    Use our local LLM to evaluate if text looks like a summary vs normal conversation.
    Returns True if it looks like a proper summary, False otherwise.
    """

    # Format the original conversation for context
    conversation_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in original_messages]
    )

    evaluation_prompt = f"""You are evaluating whether a given text is a proper summary of a conversation or just a normal conversational response.

ORIGINAL CONVERSATION:
{conversation_text}

TEXT TO EVALUATE:
{text}

A PROPER SUMMARY should:
- Concisely describe what was discussed in the conversation
- Use third person language ("The user asked about...", "The assistant recommended...")
- Cover the main topics and key information exchanged
- NOT include direct dialogue or quotes
- NOT be a conversational response

A NORMAL CONVERSATIONAL RESPONSE would:
- Use first/second person ("I", "you")
- Include dialogue or roleplay elements
- Sound like someone talking directly to the user
- Have conversational tone and structure

Please respond with only "SUMMARY" or "CONVERSATION" based on what the text appears to be."""

    # Create a separate agent instance for evaluation
    config = get_config("roleplay")
    evaluator_agent = Agent(
        config=config, model="huihui_ai/mistral-small-abliterated", verbose=False
    )

    try:
        # Get evaluation from the LLM
        response_events = list(evaluator_agent.chat_stream(evaluation_prompt))

        # Extract the text response
        evaluation_text = ""
        for event in response_events:
            if hasattr(event, "content"):
                evaluation_text += event.content

        evaluation = evaluation_text.strip().upper()

        # Check if the response contains "SUMMARY"
        return "SUMMARY" in evaluation and "CONVERSATION" not in evaluation

    except Exception as e:
        pytest.fail(f"Could not evaluate with local LLM: {e}")


@pytest.mark.integration
def test_summarization_produces_actual_summary():
    """
    Integration test that verifies summarization produces proper summaries,
    not conversational responses.
    """
    # Create agent with test config
    config = get_config("roleplay")
    agent = Agent(
        config=config, model="huihui_ai/mistral-small-abliterated", verbose=False
    )

    # Set up conversation that needs summarization
    test_messages = create_test_conversation()

    # Convert to proper message format and add to BOTH conversation histories
    for msg_data in test_messages:
        if msg_data["role"] == "user":
            message = UserMessage(role="user", content=msg_data["content"])
        else:  # assistant
            message = AgentMessage(
                role="assistant", content=msg_data["content"], tool_calls=[]
            )
        agent.conversation_history.append(message)
        agent.llm_conversation_history.append(message)  # Add to LLM history too!

    # Trigger summarization manually (keep only 2 recent messages)
    events = list(agent._auto_summarize_with_events(keep_recent=2))

    # Find the summarization finished event
    summary_event = None
    for event in events:
        if isinstance(event, SummarizationFinishedEvent):
            summary_event = event
            break

    assert summary_event is not None, "No summarization event was generated"
    assert summary_event.summary, "Summary is empty"

    print(
        f"\n--- Generated Summary ---\n{summary_event.summary}\n--- End Summary ---\n"
    )

    # Use local LLM to evaluate if this looks like a proper summary
    is_proper_summary = evaluate_with_local_llm(summary_event.summary, test_messages)

    assert is_proper_summary, (
        f"Generated text does not appear to be a proper summary. "
        f"Generated: {summary_event.summary[:200]}..."
    )


@pytest.mark.integration
def test_summarization_covers_main_topics():
    """
    Test that the summary actually covers the main topics from the conversation.
    """
    # Create agent with test config
    config = get_config("roleplay")
    agent = Agent(
        config=config, model="huihui_ai/mistral-small-abliterated", verbose=False
    )

    # Simple roleplay conversation about specific topics
    topic_messages = [
        {
            "role": "user",
            "content": "Can you roleplay as Marcus, a wise mage who teaches magic at an academy?",
        },
        {
            "role": "assistant",
            "content": 'TOOL_CALL: assume_character (call_1)\n{\n"character_name": "Marcus",\n"personality": "wise, patient, scholarly with gentle authority",\n"background": "Senior professor of elemental magic with 30 years teaching experience"\n}\nTOOL_CALL: scene_setting (call_2)\n{\n"location": "grand library of the magical academy",\n"atmosphere": "scholarly and mystical with floating books",\n"time": "afternoon study period"\n}',
        },
        {
            "role": "user",
            "content": "Professor Marcus, I'm struggling with fire magic. Can you help me?",
        },
        {
            "role": "assistant",
            "content": '"Ah, fire magic - both the most passionate and dangerous of the elements," *I look up from my ancient spellbook with kind eyes* "What specifically troubles you, young apprentice? Is it control, manifestation, or perhaps the emotional component?"\n\nTOOL_CALL: set_mood (call_3)\n{\n"mood": "patient",\n"intensity": "moderate",\n"flavor_text": "settling into his teaching role with calm authority"\n}',
        },
        {
            "role": "user",
            "content": "I can create the flames but they keep getting out of control and nearly burned down my dormitory.",
        },
        {
            "role": "assistant",
            "content": '*I nod knowingly, stroking my beard* "A classic challenge. Fire responds to emotion - fear feeds it, anger amplifies it." *I stand and gesture to an empty practice area* "The key is inner calm. Watch closely." *I extend my hand, producing a perfectly controlled flame* "Breathe with the fire, don\'t fight it. Let me teach you the Serenity Technique."\n\nTOOL_CALL: character_action (call_4)\n{\n"action": "demonstrates precise flame control with fluid hand movements"\n}\nTOOL_CALL: remember_detail (call_5)\n{\n"detail": "Student struggling with fire magic control - nearly burned dormitory, needs emotional regulation training"\n}',
        },
    ]

    # Add messages to agent
    for msg_data in topic_messages:
        if msg_data["role"] == "user":
            message = UserMessage(role="user", content=msg_data["content"])
        else:
            message = AgentMessage(
                role="assistant", content=msg_data["content"], tool_calls=[]
            )
        agent.conversation_history.append(message)
        agent.llm_conversation_history.append(message)

    # Trigger summarization
    events = list(agent._auto_summarize_with_events(keep_recent=1))

    # Find summary
    summary_event = None
    for event in events:
        if isinstance(event, SummarizationFinishedEvent):
            summary_event = event
            break

    assert summary_event is not None
    summary = summary_event.summary.lower()

    # Check that summary mentions key roleplay topics
    assert any(
        term in summary for term in ["marcus", "mage", "professor", "character"]
    ), f"Summary should mention the character Marcus or mage. Got: {summary}"
    assert any(
        term in summary for term in ["fire magic", "magic", "elemental", "flame"]
    ), f"Summary should mention fire magic or magical elements. Got: {summary}"
    assert any(
        term in summary for term in ["academy", "library", "teaching", "student"]
    ), f"Summary should mention the academy setting or teaching context. Got: {summary}"

    print(
        f"\n--- Topic Coverage Summary ---\n{summary_event.summary}\n--- End Summary ---\n"
    )


if __name__ == "__main__":
    # For manual testing
    test_summarization_produces_actual_summary()
    test_summarization_covers_main_topics()
