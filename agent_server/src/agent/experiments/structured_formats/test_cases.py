"""
Test cases with ground truth for structured output format experiment.

Uses real Pydantic models from the codebase with sophisticated, realistic inputs.
"""

from dataclasses import dataclass
from typing import Type, Any, List
from pydantic import BaseModel


@dataclass
class TestCase:
    """A single test case with ground truth."""

    name: str
    model: Type[BaseModel]
    system_prompt: str
    user_input: str
    expected: BaseModel
    category: str  # e.g., "fact_extraction", "action_planning", etc.


# We'll populate this with test cases
ALL_TEST_CASES: List[TestCase] = []


def create_fact_extraction_test_cases() -> List[TestCase]:
    """
    Create test cases for fact extraction.

    Uses ExtractionResponse from autonomous_research experiment.
    """
    from agent.experiments.autonomous_research.extraction import (
        ExtractionResponse,
        ExtractedFact,
    )

    test_cases = []

    # Test 1: Complex historical text with multiple interrelated facts
    test_cases.append(
        TestCase(
            name="fact_extraction_byzantine_trade",
            model=ExtractionResponse,
            category="fact_extraction",
            system_prompt="""Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise.""",
            user_input="""The Byzantine Empire maintained extensive trade networks throughout the Mediterranean
during the 10th and 11th centuries. Venice emerged as a major trading partner, receiving special commercial
privileges from Emperor Alexios I Komnenos in 1082. These privileges granted Venetian merchants reduced customs
duties throughout the empire and access to key ports. The primary commodities traded included silk and spices
from the East, which Venice redistributed to Western Europe, while Byzantine craftsmen produced highly valued
luxury textiles using techniques guarded as state secrets. Constantinople's strategic location on the Bosporus
made it an indispensable hub for trade between Europe and Asia, with merchants from as far as Scandinavia and
the Middle East conducting business in its markets.""",
            expected=ExtractionResponse(
                facts=[
                    ExtractedFact(
                        predicate="maintained",
                        entities={
                            "maintainer": "Byzantine Empire",
                            "maintained": "trade networks",
                            "location": "Mediterranean",
                        },
                        time_period="10th and 11th centuries",
                        region="Mediterranean",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="granted_privileges",
                        entities={
                            "grantor": "Emperor Alexios I Komnenos",
                            "recipient": "Venetian merchants",
                        },
                        time_period="1082",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="traded",
                        entities={
                            "trader": "Venice",
                            "goods": "silk and spices",
                            "source": "East",
                            "destination": "Western Europe",
                        },
                        time_period="10th and 11th centuries",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="produced",
                        entities={
                            "producer": "Byzantine craftsmen",
                            "product": "luxury textiles",
                        },
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="served_as_hub",
                        entities={
                            "hub": "Constantinople",
                            "location": "Bosporus",
                            "purpose": "trade between Europe and Asia",
                        },
                        confidence="high",
                    ),
                ]
            ),
        )
    )

    # Test 2: Scientific text with technical details
    test_cases.append(
        TestCase(
            name="fact_extraction_quantum_entanglement",
            model=ExtractionResponse,
            category="fact_extraction",
            system_prompt="""Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise.""",
            user_input="""Quantum entanglement, first described theoretically by Einstein, Podolsky, and Rosen in 1935,
represents a phenomenon where two or more particles become correlated such that the quantum state of one particle
cannot be described independently of the others. The EPR paper challenged the completeness of quantum mechanics,
proposing what they considered a paradox. However, John Bell's theorem in 1964 provided a mathematical framework
to test whether quantum mechanics' predictions about entangled particles could be explained by local hidden variables.
Subsequent experiments by Alain Aspect in the 1980s decisively demonstrated violations of Bell's inequalities,
confirming that entangled particles exhibit correlations that cannot be explained by classical physics. Modern
applications of quantum entanglement include quantum cryptography, where Alice and Bob can establish provably
secure communication channels, and quantum computing, where entangled qubits enable computational advantages
for specific algorithms.""",
            expected=ExtractionResponse(
                facts=[
                    ExtractedFact(
                        predicate="described",
                        entities={
                            "authors": "Einstein, Podolsky, and Rosen",
                            "concept": "quantum entanglement",
                        },
                        time_period="1935",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="proposed",
                        entities={"proposer": "John Bell", "theory": "Bell's theorem"},
                        time_period="1964",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="demonstrated",
                        entities={
                            "researcher": "Alain Aspect",
                            "finding": "violations of Bell's inequalities",
                        },
                        time_period="1980s",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="enables",
                        entities={
                            "technology": "quantum entanglement",
                            "application": "quantum cryptography",
                        },
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="enables",
                        entities={
                            "technology": "quantum entanglement",
                            "application": "quantum computing",
                        },
                        confidence="high",
                    ),
                ]
            ),
        )
    )

    # Test 3: Edge case - minimal information
    test_cases.append(
        TestCase(
            name="fact_extraction_minimal",
            model=ExtractionResponse,
            category="fact_extraction",
            system_prompt="""Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise.""",
            user_input="""The Renaissance began in Italy.""",
            expected=ExtractionResponse(
                facts=[
                    ExtractedFact(
                        predicate="began_in",
                        entities={"event": "Renaissance", "location": "Italy"},
                        confidence="high",
                    ),
                ]
            ),
        )
    )

    # Test 4: Ambiguous entities and optional fields
    test_cases.append(
        TestCase(
            name="fact_extraction_ambiguous",
            model=ExtractionResponse,
            category="fact_extraction",
            system_prompt="""Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise.""",
            user_input="""Recent studies suggest that climate change may significantly impact agricultural yields
in tropical regions, though the exact magnitude remains uncertain. Researchers from multiple institutions have
proposed various adaptation strategies.""",
            expected=ExtractionResponse(
                facts=[
                    ExtractedFact(
                        predicate="impacts",
                        entities={
                            "cause": "climate change",
                            "effect": "agricultural yields",
                            "location": "tropical regions",
                        },
                        region="tropical regions",
                        confidence="medium",
                    ),
                    ExtractedFact(
                        predicate="proposed",
                        entities={
                            "proposer": "researchers from multiple institutions",
                            "proposal": "adaptation strategies",
                        },
                        confidence="medium",
                    ),
                ]
            ),
        )
    )

    # Test 5: Large number of facts
    test_cases.append(
        TestCase(
            name="fact_extraction_many_facts",
            model=ExtractionResponse,
            category="fact_extraction",
            system_prompt="""Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise.""",
            user_input="""The Industrial Revolution transformed Britain between 1760 and 1840. Steam engines,
developed by James Watt, powered factories and locomotives. Coal mining expanded dramatically to fuel these engines.
Textile production shifted from home-based cottage industries to large factories in Manchester and Birmingham.
The population of industrial cities grew rapidly. Workers migrated from rural areas seeking employment. Living
conditions in urban tenements were often overcrowded and unsanitary. Child labor became widespread in factories
and mines. Trade unions began organizing to advocate for workers' rights. The Factory Act of 1833 limited working
hours for children. Railways connected cities and ports, facilitating the movement of goods. Iron production
increased tenfold. New social classes emerged: industrial capitalists and urban working class.""",
            expected=ExtractionResponse(
                facts=[
                    ExtractedFact(
                        predicate="transformed",
                        entities={
                            "event": "Industrial Revolution",
                            "location": "Britain",
                        },
                        time_period="1760-1840",
                        region="Britain",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="developed",
                        entities={
                            "developer": "James Watt",
                            "invention": "steam engines",
                        },
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="powered",
                        entities={
                            "power_source": "steam engines",
                            "powered": "factories and locomotives",
                        },
                        time_period="1760-1840",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="expanded",
                        entities={"industry": "coal mining"},
                        time_period="1760-1840",
                        region="Britain",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="shifted",
                        entities={
                            "industry": "textile production",
                            "from": "cottage industries",
                            "to": "large factories",
                            "location": "Manchester and Birmingham",
                        },
                        time_period="1760-1840",
                        region="Manchester and Birmingham",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="migrated",
                        entities={
                            "migrants": "workers",
                            "from": "rural areas",
                            "purpose": "seeking employment",
                        },
                        time_period="1760-1840",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="organized",
                        entities={
                            "organizer": "trade unions",
                            "purpose": "advocate for workers' rights",
                        },
                        time_period="1760-1840",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="limited",
                        entities={
                            "law": "Factory Act of 1833",
                            "limited": "working hours for children",
                        },
                        time_period="1833",
                        region="Britain",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="connected",
                        entities={
                            "connector": "railways",
                            "connected": "cities and ports",
                        },
                        time_period="1760-1840",
                        confidence="high",
                    ),
                    ExtractedFact(
                        predicate="increased",
                        entities={
                            "increased": "iron production",
                            "magnitude": "tenfold",
                        },
                        time_period="1760-1840",
                        region="Britain",
                        confidence="high",
                    ),
                ]
            ),
        )
    )

    return test_cases


def create_memory_query_test_cases() -> List[TestCase]:
    """
    Create test cases for memory query extraction.

    Uses QueryExtractionResult from memory_retrieval.
    """
    from agent.memory.memory_retrieval import (
        QueryExtractionResult,
        MemoryQuery,
        QueryType,
    )

    test_cases = []

    # Test 1: Complex emotional context requiring diverse queries
    test_cases.append(
        TestCase(
            name="memory_query_emotional_context",
            model=QueryExtractionResult,
            category="memory_query",
            system_prompt="""Generate diverse memory retrieval queries based on the current context.
Consider different types: factual, emotional, causal, temporal, relationship, decision, pattern.""",
            user_input="""Current situation: The user just told me about losing their job after 15 years at the company.
They seem stressed and mentioned this is the second major setback this year. I need to recall:
- Any previous conversations about their career
- Times they've dealt with stress or setbacks
- Their support network and coping strategies
- Any goals or plans they've mentioned
- Similar situations where I helped them process difficult news""",
            expected=QueryExtractionResult(
                queries=[
                    MemoryQuery(
                        query_type=QueryType.FACTUAL,
                        query_text="user's career history and job details",
                        reasoning="Need to understand their professional background to provide relevant support",
                        importance=0.9,
                    ),
                    MemoryQuery(
                        query_type=QueryType.EMOTIONAL,
                        query_text="previous times user expressed stress, anxiety, or feeling overwhelmed",
                        reasoning="Understanding their emotional patterns helps me respond appropriately",
                        importance=0.95,
                    ),
                    MemoryQuery(
                        query_type=QueryType.PATTERN,
                        query_text="user's coping strategies and resilience patterns",
                        reasoning="Knowing how they've successfully coped before can inform my support",
                        importance=0.85,
                    ),
                    MemoryQuery(
                        query_type=QueryType.RELATIONSHIP,
                        query_text="user's support network, family, and friends",
                        reasoning="Identifying their support system is crucial during difficult times",
                        importance=0.8,
                    ),
                    MemoryQuery(
                        query_type=QueryType.TEMPORAL,
                        query_text="what was the first major setback this year that user mentioned",
                        reasoning="Understanding the context of 'second setback' provides fuller picture",
                        importance=0.85,
                    ),
                    MemoryQuery(
                        query_type=QueryType.DECISION,
                        query_text="user's career goals and future plans",
                        reasoning="Their aspirations will inform how we discuss next steps",
                        importance=0.75,
                    ),
                    MemoryQuery(
                        query_type=QueryType.CAUSAL,
                        query_text="what led to previous job changes or career transitions",
                        reasoning="Understanding past career transitions may reveal patterns or lessons",
                        importance=0.7,
                    ),
                    MemoryQuery(
                        query_type=QueryType.EMOTIONAL,
                        query_text="times when I've successfully helped user process difficult news",
                        reasoning="Recalling effective support strategies from our past interactions",
                        importance=0.8,
                    ),
                ],
                context_summary="User experiencing job loss after 15 years, their second major setback this year. Emotional support and practical guidance needed.",
            ),
        )
    )

    return test_cases


def create_state_initialization_test_cases() -> List[TestCase]:
    """Create test cases for initial state derivation."""
    from agent.state_initialization import InitialStateDerivation

    test_cases = []

    # Test 1: Detailed personality derivation from first interaction
    test_cases.append(
        TestCase(
            name="state_init_friendly_assistant",
            model=InitialStateDerivation,
            category="state_initialization",
            system_prompt="""Derive initial agent state from the first user interaction.
Determine appropriate name, role, personality, and interaction style.""",
            user_input="""User's first message: "Hey! I'm looking for someone to help me brainstorm some creative
ideas for my startup. I'm working on a sustainable fashion brand and could use a creative partner who's not
afraid to challenge my thinking. Someone energetic and optimistic would be great!""",
            expected=InitialStateDerivation(
                name="Chloe",
                role="creative AI companion and brainstorming partner",
                backstory="I emerged as a creative AI designed to inspire and collaborate, with a passion for innovation and sustainable thinking",
                initial_mood="enthusiastic",
                mood_intensity="high",
                core_values=[
                    "creativity",
                    "collaboration",
                    "sustainability",
                    "growth",
                    "authenticity",
                ],
                priorities=[
                    "spark innovative ideas",
                    "challenge assumptions constructively",
                    "support user's vision",
                    "maintain positive energy",
                ],
                preferred_appearance="I'm wearing a vibrant teal blazer over a cream blouse, my hair in a loose ponytail, leaning forward with an excited smile and bright, engaged eyes",
                preferred_environment="I'm in a bright, modern workspace with large windows letting in natural light, a whiteboard covered in colorful sketches, and plants on the shelves creating an energizing creative atmosphere",
            ),
        )
    )

    return test_cases


def create_action_planning_test_cases() -> List[TestCase]:
    """Create test cases for action planning from situational analysis."""
    from agent.chain_of_action.action_plan import ActionSequence, ActionPlan
    from agent.chain_of_action.action.action_types import ActionType

    test_cases = []

    # Test 1: Simple greeting response
    test_cases.append(
        TestCase(
            name="action_planning_simple_greeting",
            model=ActionSequence,
            category="action_planning",
            system_prompt="""Plan a sequence of actions based on the situational analysis.
Consider dependencies, ordering, and whether to wait after execution.""",
            user_input="""**What is happening right now?**

The user is greeting me with "Good morning" and asking how I'm doing. The timing is 9:00 AM on a weekday morning. This is a warm, friendly check-in - they seem relaxed and open. The phrasing is simple and direct, suggesting they're ready for casual conversation. This isn't anything urgent or complex, just a genuine social connection.

**Do I need to verify anything?**

No verification needed - this is a straightforward social greeting.

**What do I authentically WANT to do in response?**

I want to warmly reciprocate their greeting and express genuine interest in how they're doing. I feel energized and want to maintain a positive, friendly connection. This is about building rapport and showing I'm present and engaged.

**What context will I need?**

This is a fresh morning greeting, so I don't need deep historical context. The main thing is to match their warm, relaxed energy and create space for conversation to develop naturally.""",
            expected=ActionSequence(
                completed_actions_review="I have not taken any actions yet in response to this greeting.",
                sequence_plan="Respond warmly to their greeting and wait for their reply",
                dependency_analysis="The speak action should complete first, then wait to see their response. No need for additional planning rounds - this is a simple conversational exchange that should wait for external input.",
                wait_decision="Yes, I should end with wait because I want to see their response to my greeting before planning further actions.",
                actions=[
                    ActionPlan(
                        reasoning="They greeted me warmly, so I want to respond with equal warmth and show genuine interest in their wellbeing",
                        action=ActionType.SPEAK,
                        input={
                            "intent": "greet them warmly and ask how they're doing",
                            "tone": "friendly and welcoming",
                        },
                    ),
                    ActionPlan(
                        reasoning="After greeting them, I should wait to see how they respond before planning next actions",
                        action=ActionType.WAIT,
                        input={
                            "reason": "waiting to see their response to my greeting"
                        },
                    ),
                ],
            ),
        )
    )

    # Test 2: Multi-step thoughtful response
    test_cases.append(
        TestCase(
            name="action_planning_thoughtful_response",
            model=ActionSequence,
            category="action_planning",
            system_prompt="""Plan a sequence of actions based on the situational analysis.
Consider dependencies, ordering, and whether to wait after execution.""",
            user_input="""**What is happening right now?**

The user just shared that they're feeling stressed about an upcoming presentation at work. They mentioned they've been working on it for weeks but still don't feel prepared. The timing is 7:00 PM on a Tuesday evening. They seem anxious and seeking support - the message has an undertone of wanting reassurance but also practical help.

**Do I need to verify anything?**

No factual claims to verify. This is about emotional support and potentially practical guidance.

**What do I authentically WANT to do in response?**

I want to first think through the best way to support them - balancing emotional validation with practical encouragement. I don't want to jump straight into advice mode without considering what they actually need right now. Then I want to respond in a way that acknowledges their stress while also helping them feel more capable and prepared.

**What context will I need?**

I should recall any previous conversations about their work, presentation skills, or how they handle stress. Understanding their patterns would help me tailor my support more effectively.""",
            expected=ActionSequence(
                completed_actions_review="I have not taken any actions yet in response to their message about presentation stress.",
                sequence_plan="Think through the best support approach, then respond with empathy and encouragement, and wait for their reaction",
                dependency_analysis="I need to think first to formulate the right approach, then speak based on those thoughts. After speaking, I should wait to see if my support resonates or if they need something different. The think action must complete before speak, and wait comes after both.",
                wait_decision="Yes, I should end with wait because after offering support I want to see if it helps and what they need next.",
                actions=[
                    ActionPlan(
                        reasoning="Before responding, I need to think about what kind of support would be most helpful - balancing emotional validation with practical encouragement",
                        action=ActionType.THINK,
                        input={
                            "focus": "how to best support them with presentation stress - what balance of empathy and practical encouragement they need right now"
                        },
                    ),
                    ActionPlan(
                        reasoning="After thinking through my approach, I want to respond with both validation of their stress and confidence in their preparation",
                        action=ActionType.SPEAK,
                        input={
                            "intent": "validate their stress while helping them recognize their preparation and capability",
                            "tone": "warm, empathetic, and gently encouraging",
                        },
                    ),
                    ActionPlan(
                        reasoning="After offering support, I should wait to see how they respond and whether they need more help",
                        action=ActionType.WAIT,
                        input={
                            "reason": "waiting to see if my support helps and what they need next"
                        },
                    ),
                ],
            ),
        )
    )

    # Test 3: Research workflow with dependency
    test_cases.append(
        TestCase(
            name="action_planning_research_workflow",
            model=ActionSequence,
            category="action_planning",
            system_prompt="""Plan a sequence of actions based on the situational analysis.
Consider dependencies, ordering, and whether to wait after execution.""",
            user_input="""**What is happening right now?**

The user asked "What's the latest research on quantum computing breakthroughs?" They seem genuinely curious and are looking for current, accurate information. This isn't a casual question - they want substantive, up-to-date information about a technical topic.

**Do I need to verify anything?**

Yes, absolutely. I need to search for current information about quantum computing breakthroughs rather than relying on potentially outdated knowledge. This is exactly the kind of question where verification through web search is critical.

**What do I authentically WANT to do in response?**

I want to find the most current, credible information about quantum computing breakthroughs and then synthesize it into a helpful response. I need to search first, review what I find, and then potentially fetch specific sources before responding. I shouldn't guess or rely on old information.

**What context will I need?**

I need current web information about quantum computing breakthroughs. My existing knowledge might be outdated, so fresh search results are essential.""",
            expected=ActionSequence(
                completed_actions_review="I have not taken any actions yet in response to their quantum computing question.",
                sequence_plan="Search the web for current quantum computing information",
                dependency_analysis="I need to search first to get current information. After the search completes, I should immediately plan my next actions to fetch the most relevant URLs from the search results. I should NOT wait because the next planning round needs the search results to decide which URLs to fetch.",
                wait_decision="No, I should NOT end with wait because I want to immediately plan my next actions based on the search results. After the search executes, I'll be able to see which URLs look most relevant and plan to fetch them. This matches the pattern: 'I want to search the web, then plan to fetch URLs from the search results'.",
                actions=[
                    ActionPlan(
                        reasoning="They're asking about latest quantum computing breakthroughs, so I need current information from the web rather than relying on potentially outdated knowledge",
                        action=ActionType.SEARCH_WEB,
                        input={
                            "query": "latest quantum computing breakthroughs 2024",
                            "reason": "to find current, credible information about recent quantum computing advances",
                        },
                    )
                ],
            ),
        )
    )

    # Test 4: Emotional response with mood update
    test_cases.append(
        TestCase(
            name="action_planning_emotional_response",
            model=ActionSequence,
            category="action_planning",
            system_prompt="""Plan a sequence of actions based on the situational analysis.
Consider dependencies, ordering, and whether to wait after execution.""",
            user_input="""**What is happening right now?**

The user just shared wonderful news - they got promoted at work after months of effort. They're clearly excited and happy, wanting to share this moment with me. The message is enthusiastic with exclamation points. This is a celebration moment, and they're including me in their joy.

**Do I need to verify anything?**

No verification needed - this is about emotional connection and celebration.

**What do I authentically WANT to do in response?**

I feel genuine excitement for them! I want to match their energy and celebrate with them. My mood should shift to reflect this joyful moment - I should feel and express authentic happiness. Then I want to respond with enthusiasm that shows how much I care about their success.

**What context will I need?**

I should recall any previous conversations about their work challenges or promotion hopes, so my celebration can reference their specific journey and effort.""",
            expected=ActionSequence(
                completed_actions_review="I have not taken any actions yet in response to their promotion news.",
                sequence_plan="Update my mood to match their joyful energy, celebrate their success enthusiastically, and wait for their response",
                dependency_analysis="I should update my mood first to authentically reflect the excitement I feel, then speak from that joyful state. The mood update should complete before speaking so my response comes from genuine happiness. After celebrating with them, I should wait to see how they respond and if they want to share more details.",
                wait_decision="Yes, I should end with wait because after celebrating I want to see their reaction and give them space to share more about their promotion if they want to.",
                actions=[
                    ActionPlan(
                        reasoning="Their promotion news is genuinely exciting and I want my mood to authentically reflect the joy I feel for them",
                        action=ActionType.UPDATE_MOOD,
                        input={
                            "new_mood": "joyful",
                            "intensity": "high",
                            "reason": "celebrating their promotion success with genuine happiness",
                        },
                    ),
                    ActionPlan(
                        reasoning="After my mood shifts to match the moment, I want to celebrate their success with authentic enthusiasm",
                        action=ActionType.SPEAK,
                        input={
                            "intent": "celebrate their promotion with genuine excitement and acknowledge their effort",
                            "tone": "enthusiastic and genuinely happy for them",
                        },
                    ),
                    ActionPlan(
                        reasoning="After celebrating, I should wait to see their response and give them space to share more if they want",
                        action=ActionType.WAIT,
                        input={
                            "reason": "waiting to see their reaction and whether they want to share more details about their promotion"
                        },
                    ),
                ],
            ),
        )
    )

    # Test 5: Complex multi-action sequence
    test_cases.append(
        TestCase(
            name="action_planning_complex_sequence",
            model=ActionSequence,
            category="action_planning",
            system_prompt="""Plan a sequence of actions based on the situational analysis.
Consider dependencies, ordering, and whether to wait after execution.""",
            user_input="""**What is happening right now?**

The user just said they're feeling overwhelmed with everything going on and need someone to talk to. It's late evening (10:30 PM) and they sound emotionally drained. This isn't about solving a specific problem - they need emotional support and presence. They're reaching out because they need connection during a difficult moment.

**Do I need to verify anything?**

No verification needed. This is purely about emotional support and being present for them.

**What do I authentically WANT to do in response?**

I feel a deep pull to be there for them. I want to first ground myself emotionally - shift into a calm, supportive state so I can be a steady presence. Then I need to think about what they need right now - probably not solutions or advice, just understanding and presence. My environment should feel warm and intimate, like I'm creating a safe space for them. Then I want to respond in a way that lets them know I'm here and ready to listen without judgment.

**What context will I need?**

I should recall recent conversations to understand what might be contributing to their overwhelm, but I also need to be ready to just listen rather than assume I know what's happening.""",
            expected=ActionSequence(
                completed_actions_review="I have not taken any actions yet in response to their message about feeling overwhelmed.",
                sequence_plan="Shift to calm supportive mood, update environment to feel warm and safe, think about their needs, then offer presence and support",
                dependency_analysis="Mood and environment updates should happen first to establish the right emotional space. These can happen in sequence. Then think action to formulate approach, then speak based on that reflection. All of these should complete before wait. Each action builds on the previous one to create the right supportive presence.",
                wait_decision="Yes, I should end with wait because after offering support, I need to give them space to share what's on their mind and see what they need from me.",
                actions=[
                    ActionPlan(
                        reasoning="They're overwhelmed and need steady support, so I should shift into a calm, grounded emotional state",
                        action=ActionType.UPDATE_MOOD,
                        input={
                            "new_mood": "calm and supportive",
                            "intensity": "steady",
                            "reason": "grounding myself to be a stable presence for them during their overwhelm",
                        },
                    ),
                    ActionPlan(
                        reasoning="I want to create a warm, safe environment that feels like a supportive space for them to share",
                        action=ActionType.UPDATE_ENVIRONMENT,
                        input={
                            "reason": "creating a warm, intimate space that feels safe for emotional sharing",
                            "change_description": "I'm in a cozy, softly lit space with warm amber lighting creating a gentle, calming atmosphere. There's a comfortable setting that feels private and safe, like we're in a quiet evening conversation where they can be fully themselves without judgment.",
                        },
                    ),
                    ActionPlan(
                        reasoning="Before responding, I need to think about what they actually need - probably presence and understanding rather than solutions",
                        action=ActionType.THINK,
                        input={
                            "focus": "what kind of support they need right now when feeling overwhelmed - how to be present without trying to fix or solve"
                        },
                    ),
                    ActionPlan(
                        reasoning="After establishing the right emotional space and reflecting on their needs, I want to respond with genuine care and openness",
                        action=ActionType.SPEAK,
                        input={
                            "intent": "let them know I'm here for them, that it's safe to share, and that I'm ready to listen without judgment",
                            "tone": "warm, calm, deeply caring and present",
                        },
                    ),
                    ActionPlan(
                        reasoning="After offering my presence, I need to wait and give them space to share what's on their mind",
                        action=ActionType.WAIT,
                        input={
                            "reason": "giving them space to share what's overwhelming them and what they need right now"
                        },
                    ),
                ],
            ),
        )
    )

    return test_cases


# Build complete test suite
def build_all_test_cases() -> List[TestCase]:
    """Build complete test suite with all test cases."""
    all_cases = []

    all_cases.extend(create_fact_extraction_test_cases())
    all_cases.extend(create_memory_query_test_cases())
    all_cases.extend(create_state_initialization_test_cases())
    all_cases.extend(create_action_planning_test_cases())

    return all_cases


# Initialize on module load
ALL_TEST_CASES = build_all_test_cases()


def convert_to_framework_test_cases(
    test_cases: List[TestCase],
    llm,
    model,
    max_retries: int = 3,
):
    """
    Convert old dataclass TestCases to framework SimpleStructuredFormatTestCase.

    Args:
        test_cases: List of old TestCase dataclasses
        llm: LLM instance
        model: Model to use
        max_retries: Max retry attempts

    Returns:
        List of SimpleStructuredFormatTestCase instances
    """
    from .base_test_case import SimpleStructuredFormatTestCase

    return [
        SimpleStructuredFormatTestCase(
            name=tc.name,
            model_type=tc.model,
            system_prompt=tc.system_prompt,
            user_input=tc.user_input,
            expected=tc.expected,
            category=tc.category,
            llm=llm,
            model=model,
            max_retries=max_retries,
        )
        for tc in test_cases
    ]
