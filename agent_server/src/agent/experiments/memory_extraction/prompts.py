"""Prompts for memory extraction approaches."""

# Approach A: Fact list extraction
FACT_LIST_PROMPT = """Extract all facts from this memory as individual, searchable statements.

MEMORY:
{content}

For each fact:
- Make it a standalone, searchable statement
- Be specific and concrete
- Include relevant entities (people, places, things)
- Assign a fact type (preference, event, relationship, state, fact, question, appearance, environment, emotion, value)
- Rate your confidence from 0.0 to 1.0

Do NOT include:
- Filler words or conversational padding
- Hedging language ("kind of", "sort of", "maybe")
- Redundant restatements
- Timestamps that are in the metadata

Extract all important facts."""


# Approach B: Structured extraction by category
STRUCTURED_PROMPT = """Extract information from this memory organized by category.

MEMORY:
{content}

Extract information in these categories:
1. PEOPLE: Facts about people mentioned and what we learn about them
2. EVENTS: Events or plans discussed (past, present, or future)
3. PREFERENCES: Preferences, opinions, or likes/dislikes expressed
4. EMOTIONS: Emotional states, moods, or feelings
5. QUESTIONS: Questions asked or answered

For each category, list the relevant facts as clear, searchable statements.
Also provide a one-sentence summary of the memory."""


# Approach C: Query-focused extraction
QUERY_FOCUSED_PROMPT = """Analyze this memory and identify what questions it could answer.

MEMORY:
{content}

For each potential question:
1. State the question clearly
2. Provide a concise answer based on the memory content

Focus on questions someone might actually ask when searching their memories, like:
- "What does [person] like?"
- "What happened when [event]?"
- "How does [person] feel about [topic]?"
- "What is [person] wearing?"

Also provide a one-sentence summary of the memory."""


# Approach D: Entity-centric extraction
ENTITY_CENTRIC_PROMPT = """Extract information organized by the people and entities mentioned.

MEMORY:
{content}

For each person or significant entity mentioned:
1. ENTITY NAME: The name of the person/entity
2. LEARNED FACTS: What do we learn about them? (traits, preferences, states)
3. ACTIONS: What did they say or do?
4. RELATIONSHIPS: How do they relate to others mentioned?

Also provide a one-sentence summary of the memory."""


# Approach E: Minimal extraction
MINIMAL_PROMPT = """Identify the single most important fact in this memory.

MEMORY:
{content}

What is the ONE most important, memorable fact from this memory?
Answer in 10 words or less.
Also identify the type of fact (preference, event, relationship, state, fact, appearance, environment, emotion, value)."""


# Approach for high compression
HIGH_COMPRESSION_PROMPT = """Extract only the 3 most important facts from this memory.

MEMORY:
{content}

Identify the 3 most essential, memorable facts. Each should be:
- A standalone, searchable statement
- Specific and concrete
- Include relevant entities

Prioritize facts that would be most useful for future retrieval."""


# Approach for low compression (comprehensive)
LOW_COMPRESSION_PROMPT = """Extract ALL facts comprehensively from this memory.

MEMORY:
{content}

Be thorough and extract every piece of information, including:
- Main facts and details
- Background information
- Descriptions of appearance, environment, mood
- Relationships and interactions
- Preferences and opinions
- Actions and events

Do not summarize or combine facts - extract each as a separate, searchable statement."""


# Annotation prompt
ANNOTATION_PROMPT = """Evaluate whether this extracted fact is accurate based on the original memory.

ORIGINAL MEMORY:
{original}

EXTRACTED FACT:
{fact}

Determine if this fact is:
- CORRECT: The fact is accurately stated and directly supported by the original
- HALLUCINATED: The fact is not supported by or contradicts the original
- INFERRED: The fact is a reasonable inference but not explicitly stated

Provide your label and brief reasoning."""


# Omissions prompt
OMISSIONS_PROMPT = """Identify important facts from the original memory that were NOT extracted.

ORIGINAL MEMORY:
{original}

EXTRACTED FACTS:
{extracted_facts}

List any important facts from the original memory that are missing from the extracted facts.
Focus on facts that would be useful for future retrieval."""


# Test query generation prompt
GENERATE_TEST_QUERIES_PROMPT = """Given this memory content, generate 2-3 questions that someone might ask that this memory would answer.

MEMORY:
{content}

Generate natural questions that:
- Could be answered by information in this memory
- Are specific enough to distinguish this memory from others
- Cover different types of information (appearance, events, relationships, preferences, etc.)

Return only the questions, nothing else."""
