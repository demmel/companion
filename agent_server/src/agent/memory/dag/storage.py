from pathlib import Path
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.memory.dag.action_log import MemoryActionLog
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.timeit import timeit


def save_dag_memory(
    dir: Path,
    prefix: str,
    dag_memory_manager: DagMemoryManager,
) -> None:
    with timeit("Saving DAG memory to file"):
        dag_memory_manager.save_to_file(_dag_file_name(dir, prefix))
    with timeit("Saving DAG memory action log to file"):
        dag_memory_manager.save_action_log(_dag_action_log_file_name(dir, prefix))


def load_dag_memory(
    dir: Path,
    prefix: str,
    trigger_history: TriggerHistory,
    use_individual_formatting: bool,
    resave: bool = False,
) -> DagMemoryManager:
    with timeit("Loading DAG memory from file"):
        dag = DagMemoryManager.load_from_file(
            _dag_file_name(dir, prefix), trigger_history, use_individual_formatting
        )
    # with timeit("Loading DAG memory from action log file"):
    #     dag = DagMemoryManager.load_from_action_log(
    #         self._dag_action_log_file_name(prefix), trigger_history=trigger_history
    #     )
    with timeit("Loading DAG memory action log from file"):
        action_log = MemoryActionLog.load_from_file(
            _dag_action_log_file_name(dir, prefix)
        )
    dag.action_log = action_log
    with timeit("Replaying DAG memory action log"):
        _, _ = dag.action_log.replay_from_empty(trigger_history)

    if resave:
        save_dag_memory(
            dir,
            prefix,
            dag,
        )

    return dag


def _dag_file_name(dir: Path, prefix: str) -> str:
    """Get the DAG memory file name for a conversation"""
    return f"{dir}/{prefix}_dag.json"


def _dag_action_log_file_name(dir: Path, prefix: str) -> str:
    """Get the DAG memory action log file name for a conversation"""
    return f"{dir}/{prefix}_dag_actions.json"
