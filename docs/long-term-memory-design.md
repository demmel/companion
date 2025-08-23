# Long-Term Memory System Design

## Overview

The agent needs a sophisticated long-term memory system that persists across conversations and provides intelligent retrieval of relevant memories based on multiple similarity dimensions. These memories will be the basis for the agent's context in order to eliminate summarization and long stream of consciousness to effectively utilize the context window while maintaining coherence in conversation.

## Memory Types & Retrieval Strategies

### 1. Conceptual Similarity

These are memories that share similar themes, topics, or ideas. They are retrieved based on semantic meaning and contextual relevance. The agent may analyze the current situation and user input to identify relevant memories based on conceptual overlap. The mechanism for determining conceptual similarity could involve techniques such as embedding-based similarity search, where both the query and memories are represented in a high-dimensional space.

### 2. Temporal Similarity

These are memories that happened in a similar time frame. They are retrieved based on the timing of events and the user's interaction history. The agent may use techniques such as time-based indexing to quickly locate relevant memories from specific periods. This can help in recalling past events or conversations that are contextually linked by time.

This is useful for:

- retrieving the most recent interactions or events that may be relevant to the current conversation
- recalling past events or conversations that are brought up or linked to other memories that are currently recalled

### 3. Contextual Similarity

These are memories that were created while other memories were being recalled. They're related just by the fact that the agent had some memories in its context while the memory was formed. In a sense, temporal similarity is a subset of contextual similarity, as the timing of events is often influenced by the surrounding context.

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

## Implementation

To begin, we'll probably implement the simplest system possible. Memories will be stored in a simple JSON blob format, with each memory containing its relevant metadata and content. This will allow for easy serialization and deserialization, making it straightforward to save and load memories as needed.

We'll start with simple persistence and static storage, keeping all memories indefinitely until we implement more advanced pruning strategies.

Our first retrieval strategy will be temporal similarity, focusing on memories that are contextually linked by time. This is similar the current conversation history and recent interactions. However, it will afford the agent to the ability to recall past events or conversations that are contextually linked to the user's query.
