# Reasoning System Integration Plan

## Overview
Replace the current LLM-direct-output agent architecture with a reasoning-driven system that enables self-reflection and iterative response building.

## Current vs New Architecture

### Current Flow
```
User Input → LLM Response → Parse Tools → Execute Tools → Done
```

### New Flow  
```
User Input → [Iterative Reasoning Loop] → End Turn

Iterative Loop:
1. Reasoning (about current context) → May decide to end turn
2. Tool Calls (if reasoning suggests them)
3. Tool Results  
4. Agent Response (using reasoning + tool results)
5. Go back to step 1, but now reasoning about agent's own response
```

## Two Types of Reasoning Context

### 1. Reasoning About User Input
- What is the user trying to communicate?
- What emotions/intentions/needs are present?
- How does this connect to our conversation history?
- What should I do in response?

### 2. Reasoning About Agent's Own Response
- Did I fully address what they needed?
- Should I add more context, visuals, or information?
- Are there important details I should remember?
- Is this response complete or should I continue?

## Message Structure

Each iteration produces structured content in order:
- **ThoughtContent** (reasoning analysis as JSON)
- **ToolCallContent** (tool executions based on reasoning)  
- **TextContent** (response informed by reasoning + tool results)

## Core Agent Integration

### Replace in `chat_stream()` method:

**Old iteration loop:**
```python
for iteration in range(1, max_iterations + 1):
    # Build LLM prompt
    # Get streaming LLM response  
    # Parse for text/thoughts and tool calls
    if not tool_events or is_final_iteration:
        break  # End turn
    # Execute tools, add results, continue iteration
```

**New iteration loop:**
```python
for iteration in range(1, max_iterations + 1):
    # Reasoning step (about user input OR agent's previous response)
    reasoning_result = reason_about_text(
        text=current_context_to_analyze,
        message_role="user" if first_iteration else "assistant", 
        conversation_context=full_conversation,
        tool_registry=self.tools,
        llm=self.llm,
        model=self.model
    )
    
    # Stream the full reasoning as JSON for debugging
    reasoning_json = reasoning_result.model_dump_json(indent=2)
    yield AgentTextEvent(text=reasoning_json, type="thought")
    
    # Check if reasoning says we're done (no tools proposed)
    if not reasoning_result.proposed_tools:
        break
    
    # Execute tools from reasoning
    for proposed_tool in reasoning_result.proposed_tools:
        yield ToolStartedEvent(
            tool_name=proposed_tool.tool_name,
            tool_id=f"reasoning_{iteration}_{tool_index}",
            parameters=proposed_tool.parameters
        )
        
        result = self.tools.execute(...)
        
        yield ToolFinishedEvent(
            tool_name=proposed_tool.tool_name,
            tool_id=tool_id,
            parameters=proposed_tool.parameters,
            result=result
        )
    
    # Generate response using LLM (informed by reasoning + tool results)
    # TODO: Implement response generation based on reasoning and tool results
    response_text = generate_response_from_reasoning_and_tools(...)
    yield AgentTextEvent(text=response_text, type="text")
    
    # Update context for next iteration (agent will reason about this response)
    current_context_to_analyze = response_text
```

## Key Changes

### What We're Replacing
1. **LLM direct tool output** → **Reasoning-decided tool calls**
2. **Simple termination logic** (`no tools = done`) → **Reasoning-driven termination**
3. **Single response per turn** → **Iterative response building**
4. **No self-reflection** → **Agent reasons about its own responses**

### What We're Preserving
- `chat_stream()` method signature and streaming interface
- Tool execution infrastructure (`ToolRegistry.execute()`)
- Event types (`AgentTextEvent`, `ToolStartedEvent`, `ToolFinishedEvent`)
- Conversation history storage patterns
- Max iteration safety limits
- Auto-summarization logic

### What We're Adding
- **Structured reasoning analysis** before any action
- **Self-reflection capability** (reasoning about own responses)
- **Full reasoning JSON streaming** for debugging and transparency
- **Iterative enhancement** within single conversation turns
- **Understanding + Acting** explicit separation

## Implementation Status

### Completed
- ✅ Basic reasoning function with structured output (`reason_about_text`)
- ✅ Pydantic models for reasoning results (`ReasoningResult`, `ProposedToolCall`)
- ✅ Tool registry integration (uses actual tool descriptions)
- ✅ New message types (`ReasoningThoughtContent`, `ToolCallContent`)
- ✅ Testing with realistic conversation data

### Next Steps
1. **Integrate reasoning loop into core agent `chat_stream()` method**
2. **Implement response generation using reasoning + tool results**
3. **Add reasoning-driven termination logic**
4. **Test with Chloe's conversation data**
5. **Handle context switching between user input and agent response analysis**

## Benefits for Chloe

This architecture will enable Chloe to:
- **Understand deeply** before acting (not just pattern-match to tools)
- **Self-reflect** on her responses to improve them iteratively
- **Build continuity** by reasoning about conversation history
- **Make thoughtful decisions** about when to remember details, change appearance, etc.
- **Provide transparency** through visible reasoning process for debugging

The reasoning JSON will show exactly how she's interpreting interactions, what she considers important, and why she makes specific choices - crucial for building a meaningful companion relationship.