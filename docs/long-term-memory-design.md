# Long-Term Memory System Design

## Overview

The agent needs a sophisticated long-term memory system that persists across conversations and provides intelligent retrieval of relevant memories based on multiple similarity dimensions. This system would be triggered by `ADD_MEMORY` and `REMOVE_MEMORY` actions that the agent can consciously choose to use.

## Memory Types & Retrieval Strategies

### 1. Conceptual Similarity (Vector-based)
- **Implementation**: Vector database (embeddings)
- **Purpose**: Retrieve memories with similar semantic meaning
- **Example**: User mentions "debugging Python" → retrieves memories about coding challenges, error handling, programming help
- **Technology**: Sentence transformers, vector similarity search

### 2. Temporal Similarity (Time-based)
- **Implementation**: Time-locality indexing
- **Purpose**: Retrieve memories from similar time periods
- **Example**: "Around this time last week/month" type queries
- **Use cases**: 
  - Seasonal patterns ("Last winter I was feeling...")
  - Recurring events ("User usually asks about work on Monday mornings")
  - Temporal context ("What was happening when we last discussed this?")

### 3. Contextual Similarity (Context-aware)
- **Implementation**: Context embedding at memory creation time
- **Purpose**: Retrieve memories created in similar conversational contexts
- **Example**: 
  - Context: User is frustrated + asking technical questions → retrieves memories of similar emotional/topic combinations
  - Context: Evening conversation + personal topics → retrieves intimate/reflective memories
- **Factors**: User mood, conversation topic, time of day, recent interaction patterns

## Core Challenges

### Memory Retrieval
- **Relevance Scoring**: How to rank memories across different similarity types
- **Multi-dimensional Search**: Combining conceptual + temporal + contextual signals
- **Context Budget**: How many memories can fit in prompt context without overwhelming?
- **Dynamic Weighting**: Should temporal vs conceptual vs contextual weights change based on situation?

### Memory Management Over Time
Several open questions need resolution:

#### Persistence Strategy
- **Simple Persistence**: Memories stay forever
- **Time-to-Live (TTL)**: Memories expire after X days/months
- **Importance Scoring**: Keep important memories longer
- **Access-based**: Frequently recalled memories stay longer

#### Memory Evolution
- **Static Storage**: Memories never change once created
- **Memory Merging**: Similar memories combine over time
- **Memory Updating**: Memories get refined/corrected with new information
- **Memory Compression**: Old memories get summarized to save space

#### Memory Pruning
- **Capacity Limits**: Max N memories total
- **Relevance Pruning**: Remove least-accessed memories
- **Redundancy Detection**: Remove near-duplicate memories
- **Importance Thresholds**: Only keep memories above certain significance

## Technical Architecture

### Storage Layer
- **Vector Database**: For semantic similarity (ChromaDB, Pinecone, etc.)
- **Time Series Database**: For temporal queries
- **Relational Database**: For metadata, relationships, context info
- **File System**: For large memory content (if needed)

### Retrieval Pipeline
1. **Query Analysis**: Determine what type of memory retrieval is needed
2. **Multi-Search**: Run parallel searches across vector/temporal/contextual indices
3. **Ranking & Fusion**: Combine results using learned or rule-based weights
4. **Context Fitting**: Select top N memories that fit in available context budget
5. **Formatting**: Present memories in appropriate format for agent consumption

### Integration Points
- **Action System**: `ADD_MEMORY` action creates memories with full context
- **Trigger System**: Each trigger can potentially activate memory retrieval
- **Prompt System**: Retrieved memories get injected into relevant prompts
- **State System**: Memories might influence agent state and behavior

## Memory Content Structure

### Memory Record
```python
class Memory:
    id: str
    content: str                    # The actual memory text
    created_at: datetime           # When memory was created
    last_accessed: datetime        # For access-based management
    importance_score: float        # Agent's assessment of importance
    
    # Context at creation time
    conversation_context: str      # What was happening when created
    emotional_context: str         # Agent's mood/feelings
    user_context: str             # User's apparent mood/situation
    topic_tags: List[str]         # Extracted topics/themes
    
    # For retrieval
    embedding: List[float]         # Vector representation
    temporal_features: dict        # Time patterns, seasonality
    contextual_features: dict      # Situational similarity features
```

### Memory Triggers
- **Explicit**: Agent consciously decides to remember something
- **Automatic**: System suggests memories based on conversation patterns
- **Query-based**: User asks "Do you remember when...?"
- **Contextual**: Similar situation triggers related memory retrieval

## Implementation Phases

### Phase 1: Basic Memory Actions
- Implement `ADD_MEMORY` and `REMOVE_MEMORY` actions
- Simple storage in JSON files or local database
- Manual memory retrieval for testing

### Phase 2: Single Retrieval Type
- Implement one retrieval strategy (likely conceptual/vector-based)
- Basic integration with prompt system
- Simple relevance ranking

### Phase 3: Multi-dimensional Retrieval
- Add temporal and contextual similarity
- Implement ranking fusion algorithm
- Context budget management

### Phase 4: Memory Management
- Implement chosen persistence strategy
- Add memory lifecycle management
- Performance optimization

### Phase 5: Advanced Features
- Memory merging/updating capabilities
- Learning-based relevance weighting
- Cross-conversation memory analytics

## Open Research Questions

1. **Retrieval Balance**: What's the optimal mix of conceptual/temporal/contextual weighting?
2. **Memory Granularity**: Should memories be full conversations, individual exchanges, or specific facts?
3. **Privacy & Control**: How should users control what gets remembered?
4. **Memory Accuracy**: How to handle misremembered or outdated information?
5. **Emotional Memory**: Should emotional significance boost memory importance?
6. **Social Memory**: How to handle memories about multiple users?
7. **Memory Sharing**: Should memories be shareable between agent instances?

## Success Metrics

- **Relevance**: Retrieved memories are actually useful for current context
- **Coverage**: Important past interactions can be successfully recalled
- **Performance**: Memory retrieval doesn't significantly slow response time  
- **Storage Efficiency**: Memory system scales with conversation history
- **User Experience**: Memory feels natural and helpful, not overwhelming

## Related Work

- Personal AI assistants (Replika, Character.AI memory systems)
- Conversational AI memory (Google LaMDA, Anthropic Constitutional AI)
- Vector databases for AI (Pinecone, Weaviate, ChromaDB)
- Temporal memory systems in cognitive architectures
- Human memory models (episodic vs semantic memory)