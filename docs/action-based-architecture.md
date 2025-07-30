# Action-Based Architecture Design

## Overview

This document outlines the new action-based architecture for the agent system, replacing the monolithic reasoning loop with a modular, scalable approach that separates concerns and provides better debugging visibility.

## Core Principles

1. **Separation of Concerns**: Each action has a specific responsibility
2. **Modularity**: Actions can be added/modified without touching existing code
3. **Flexibility**: Agent decides whether to speak or not, no forced constraints
4. **Rich Context**: Actions build their own context from fresh state as needed
5. **Proper Events**: Different action types emit appropriate event types for UI

## Action Types

### Cognitive Actions
- `THINK`: Process emotional reactions and analyze the situation

### State Management Actions
- `UPDATE_MOOD`: Change the agent's current mood and intensity
- `ADD_MEMORY`: Store important details
- `REMOVE_MEMORY`: Forget specific memories
- `UPDATE_APPEARANCE`: Visual changes
- `UPDATE_ENVIRONMENT`: Setting changes
- `ADD_GOAL`: Add new goals
- `REMOVE_GOAL`: Complete/abandon goals
- `ADD_DESIRE`: New immediate wants
- `REMOVE_DESIRE`: Satisfy/abandon desires

### Communication Actions
- `SPEAK`: Generate a conversational response (optional!)

### Meta Actions
- `DONE`: Complete sequence

## Architecture Components

### 1. Trigger System

```python
class TriggerType(str, Enum):
    USER_INPUT = "user_input"
    # Future: TOOL_RESULT, SELF_REFLECTION, TIMER_BASED, etc.

class TriggerEvent(BaseModel):
    trigger_type: TriggerType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class UserInputTrigger(TriggerEvent):
    trigger_type: TriggerType = TriggerType.USER_INPUT
```

### 2. Execution Context

Minimal context that carries immutable information. Actions build what they need on-demand.

```python
class ExecutionContext(BaseModel):
    trigger: TriggerEvent
    completed_actions: List[ActionResult] = Field(default_factory=list)
    session_id: str
    
    def add_completed_action(self, result: ActionResult):
        self.completed_actions.append(result)
```

### 3. Action Classes (Distributed Design)

Each action class defines both what it does and what context it needs:

```python
class BaseAction:
    action_type: ActionType
    
    @classmethod
    def get_action_description(cls) -> str:
        """What this action does"""
        raise NotImplementedError
        
    @classmethod
    def get_context_description(cls) -> str:
        """What context this action needs when planned"""
        raise NotImplementedError
    
    @classmethod
    def create_executor(cls, state: State, conversation_history: ConversationHistory, llm: LLM, model: SupportedModel) -> 'BaseActionExecutor':
        """Factory for action executor"""
        raise NotImplementedError

class ThinkAction(BaseAction):
    action_type = ActionType.THINK
    
    @classmethod
    def get_action_description(cls) -> str:
        return "Process emotional reactions and analyze the situation"
        
    @classmethod
    def get_context_description(cls) -> str:
        return "Specific aspects to focus thinking on - emotional elements, relationship dynamics, or particular details requiring analysis"
    
    @classmethod
    def create_executor(cls, state: State, conversation_history: ConversationHistory, llm: LLM, model: SupportedModel) -> 'ThinkActionExecutor':
        return ThinkActionExecutor(state, conversation_history, llm, model)
```

### 4. Action Executors

Actions build their own context from fresh state and have full execution logic:

```python
class BaseActionExecutor:
    def __init__(self, state: State, conversation_history: ConversationHistory, llm: LLM, model: SupportedModel):
        self.state = state
        self.conversation_history = conversation_history
        self.llm = llm
        self.model = model
    
    def serialize_conversation_history(self) -> str:
        """Serialize when needed for prompts"""
        return self.conversation_history.get_summarized_history_as_string()
    
    def build_agent_state_description(self) -> str:
        """Build fresh state description when needed"""
        return build_agent_state_description(self.state)

class ThinkActionExecutor(BaseActionExecutor):
    def execute(self, action_plan: ActionPlan, context: ExecutionContext) -> ActionResult:
        state_desc = self.build_agent_state_description()  # Fresh state
        history_str = self.serialize_conversation_history()  # When needed
        
        prompt = f"""I am {self.state.name}, processing what just happened.
        
{state_desc}

Conversation history:
{history_str}

What happened: {context.trigger.content}
Focus: {action_plan.context}

My thoughts:"""
        
        result = self.llm.generate_complete(self.model, prompt)
        return ActionResult(action=ActionType.THINK, result_summary=result, ...)
```

### 5. Action Planning

```python
class ActionPlan(BaseModel):
    action: ActionType
    context: str = Field(description="Situational details this action should focus on")

class ActionSequence(BaseModel):
    actions: List[ActionPlan] = Field(description="Actions in execution order")
    can_extend: bool = Field(description="Whether more actions can be added on-the-fly")
    reasoning: str = Field(description="Why this sequence was chosen")

class ActionPlanner:
    def build_planning_prompt(self, context: ExecutionContext) -> str:
        action_guidance = ActionRegistry.get_planning_guidance()
        
        return f"""I am {self.state.name}, deciding what actions to take...

Available actions:
{action_guidance}

For each action I plan, I need to provide specific context about what that action should focus on in this situation.

What should I do next?"""
```

### 6. Action Registry

```python
class ActionRegistry:
    ACTIONS = {
        ActionType.THINK: ThinkAction,
        ActionType.UPDATE_MOOD: UpdateMoodAction,
        ActionType.SPEAK: SpeakAction,
        ActionType.ADD_MEMORY: AddMemoryAction,
        # ... etc
    }
    
    @classmethod
    def get_planning_guidance(cls) -> str:
        """Build complete action info for planner prompt"""
        guidance = []
        for action_type, action_class in cls.ACTIONS.items():
            action_desc = action_class.get_action_description()
            context_desc = action_class.get_context_description()
            guidance.append(f"- {action_type.value}: {action_desc}")
            guidance.append(f"  Context needed: {context_desc}")
        return "\n".join(guidance)
```

### 7. Event Emission

Different action types emit appropriate events for UI treatment:

```python
class ActionExecutor:
    def execute_action(self, action_plan: ActionPlan, context: ExecutionContext) -> Iterator[AgentEvent]:
        # State-modifying actions get tool events for UI
        if action_plan.action in [ActionType.UPDATE_MOOD, ActionType.ADD_MEMORY, ...]:
            yield ToolStartedEvent(
                tool_name=action_plan.action.value,
                tool_id=f"action_{action_plan.action.value}_{timestamp}",
                parameters={"context": action_plan.context}
            )
        
        # Execute the action logic (ALL actions have execution)
        result = self._execute_action_logic(action_plan, context)
        
        # Emit appropriate finish events
        if action_plan.action in [ActionType.UPDATE_MOOD, ActionType.ADD_MEMORY, ...]:
            yield ToolFinishedEvent(
                tool_id=f"action_{action_plan.action.value}_{timestamp}",
                result=result.result_summary
            )
        elif action_plan.action == ActionType.THINK:
            yield AgentTextEvent(content=result.result_summary, is_thought=True)
        elif action_plan.action == ActionType.SPEAK:
            yield AgentTextEvent(content=result.result_summary, is_thought=False)
```

### 8. Repetition Analysis

Prevents overthinking loops by analyzing action patterns:

```python
class RepetitionAnalyzer:
    def analyze(self, completed_actions: List[ActionResult], proposed_action: ActionType, trigger_context: ExecutionContext) -> RepetitionAnalysis:
        recent_action_types = [a.action for a in completed_actions[-5:]]
        # Analyze for repetitive patterns and recommend stopping if needed
        
class ActionSystemEvent(AgentEvent):
    """System-level action events (not user-facing content)"""
    event_type: str
    message: str
    action_sequence_id: str

# Usage in main loop
if repetition_analysis.should_stop:
    yield ActionSystemEvent(
        event_type="repetition_stopped",
        message=f"Stopped due to repetition: {repetition_analysis.pattern_detected}",
        action_sequence_id=context.session_id
    )
    break
```

## Main Execution Flow

```python
class ActionBasedReasoningLoop:
    def process_trigger(self, user_input: str, conversation_history: ConversationHistory) -> Iterator[AgentEvent]:
        # 1. Create trigger
        trigger = UserInputTrigger(content=user_input)
        context = ExecutionContext(trigger=trigger)
        
        # 2. Plan initial actions
        action_sequence = self.planner.plan_initial_sequence(context)
        
        # 3. Execute actions with repetition checking
        for action_plan in action_sequence.actions:
            # Check for repetition
            if len(context.completed_actions) >= 2:
                repetition_analysis = self.repetition_analyzer.analyze(
                    context.completed_actions, action_plan.action, context
                )
                if repetition_analysis.should_stop:
                    yield ActionSystemEvent(
                        event_type="repetition_stopped",
                        message=f"Stopped: {repetition_analysis.pattern_detected}",
                        action_sequence_id=context.session_id
                    )
                    break
            
            # Execute action
            executor = ActionRegistry.ACTIONS[action_plan.action].create_executor(
                self.state, conversation_history, self.llm, self.model
            )
            
            for event in self.action_executor.execute_action(action_plan, context, executor):
                yield event
            
            # Check for extensions
            if action_sequence.can_extend:
                extensions = self._check_for_extensions(context)
                action_sequence.actions.extend(extensions)
```

## Integration with Current System

### Minimal Agent Changes

```python
class Agent:
    def __init__(self, model, llm, auto_save=True):
        # ... existing initialization ...
        self.action_reasoning_loop = ActionBasedReasoningLoop(llm, model, self.tools, self.state)
    
    def chat_stream(self, user_input: str) -> Iterator[AgentEvent]:
        # ... existing character configuration logic ...
        
        # Replace existing reasoning loop
        for event in self.action_reasoning_loop.process_trigger(user_input, self.conversation_history):
            yield event
            
        # ... existing cleanup logic ...
```

### Backward Compatibility

- Events map to existing `AgentEvent` types
- State updates work through existing state management
- Tool-like events for state modifications provide proper UI feedback
- No changes needed to frontend event handling

## Benefits

1. **Modularity**: Each action is self-contained and independently testable
2. **Flexibility**: Agent can choose any combination of actions, including not speaking
3. **Rich Context**: Actions build fresh context from current state as needed
4. **Performance**: Multi-action planning reduces LLM calls
5. **Debugging**: Clear visibility into what each action does and why
6. **Scalability**: Easy to add new action types without touching existing code
7. **UI Integration**: Proper events for different action types

## Migration Strategy

1. **Phase 1**: Implement core architecture alongside current system
2. **Phase 2**: A/B test action-based vs current reasoning on simple interactions
3. **Phase 3**: Gradually migrate action types (start with THINK/SPEAK)
4. **Phase 4**: Full migration with all action types and rich prompts
5. **Phase 5**: Add advanced features (multi-action planning, complex triggers)

## Future Enhancements

- **Complex Triggers**: Tool results, self-reflection, timer-based actions
- **Action Dependencies**: Actions that must complete before others can start
- **Parallel Actions**: Actions that can run simultaneously
- **Action Chains**: Predefined sequences for common patterns
- **Learning**: Optimize action selection based on success patterns