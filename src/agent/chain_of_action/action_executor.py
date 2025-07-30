"""
Action executor for running action sequences.
"""

import logging
from typing import List, TYPE_CHECKING

from .action_plan import ActionSequence
from .action_registry import ActionRegistry
from .context import ExecutionContext, ActionResult

if TYPE_CHECKING:
    from agent.state import State
    from agent.conversation_history import ConversationHistory
    from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes action sequences and manages execution context"""
    
    def __init__(self, registry: ActionRegistry):
        self.registry = registry
    
    def execute_sequence(self, sequence: ActionSequence, context: ExecutionContext,
                        state: 'State', conversation_history: 'ConversationHistory',
                        llm: 'LLM', model: 'SupportedModel') -> List[ActionResult]:
        """Execute a complete action sequence"""
        
        results = []
        
        logger.debug(f"=== EXECUTING ACTION SEQUENCE ===")
        logger.debug(f"ACTIONS: {len(sequence.actions)}")
        logger.debug(f"PRIOR COMPLETED: {len(context.completed_actions)}")
        
        # Execute each action in sequence
        for i, action_plan in enumerate(sequence.actions):
            logger.debug(f"Executing action {i+1}/{len(sequence.actions)}: {action_plan.action.value}")
            
            try:
                # Create action instance
                action = self.registry.create_action(action_plan.action)
                
                # Execute the action
                result = action.execute(
                    action_plan=action_plan,
                    context=context,
                    state=state,
                    conversation_history=conversation_history,
                    llm=llm,
                    model=model
                )
                
                # Add result to list
                results.append(result)
                
                # Update context with completed action
                context.completed_actions.append(result)
                
                logger.debug(f"Action completed: {result.action.value} ({'success' if result.success else 'failed'})")
                if result.result_summary:
                    logger.debug(f"Result: {result.result_summary[:100]}...")
                
            except Exception as e:
                # Catastrophic failure - action couldn't be created or executed
                logger.error(f"Catastrophic failure executing {action_plan.action.value}: {e}", exc_info=True)
                
                # Create error result
                error_result = ActionResult(
                    action=action_plan.action,
                    result_summary="",
                    context_given=action_plan.context,
                    duration_ms=0.0,
                    success=False,
                    error=f"Execution exception: {str(e)}"
                )
                
                results.append(error_result)
                context.completed_actions.append(error_result)
                
                # Stop execution - this is a serious system failure
                logger.error(f"Stopping sequence execution due to catastrophic failure")
                break
        
        logger.debug(f"=== SEQUENCE EXECUTION COMPLETE ===")
        logger.debug(f"Total actions: {len(results)}")
        successful = sum(1 for r in results if r.success)
        logger.debug(f"Successful: {successful}/{len(results)}")
        
        return results