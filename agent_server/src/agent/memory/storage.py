from enum import Enum
from pathlib import Path
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.memory.dag.dag_memory_manager import DagMemoryManager
from agent.memory.memory import IMemory


def save_memory(
    dir: Path,
    prefix: str,
    memory: IMemory,
) -> None:
    match memory:
        case DagMemoryManager():
            from agent.memory.dag.storage import save_dag_memory

            save_dag_memory(dir, prefix, memory)
        case _:
            raise ValueError(f"Unsupported memory type: {type(memory)}")


def load_memory(
    dir: Path,
    prefix: str,
    trigger_history: TriggerHistory,
    resave: bool = False,
) -> IMemory:
    memory_type = infer_memory_type_from_files(dir, prefix)
    match memory_type:
        case MemoryType.DAG:
            from agent.memory.dag.storage import load_dag_memory

            return load_dag_memory(
                dir,
                prefix,
                trigger_history,
                resave=resave,
            )
        case _:
            raise ValueError(f"Unsupported memory type: {memory_type}")


class MemoryType(str, Enum):
    DAG = "dag"


def infer_memory_type_from_files(
    dir: Path,
    prefix: str,
) -> MemoryType:
    from agent.memory.dag.storage import _dag_file_name

    dag_file = Path(_dag_file_name(dir, prefix))
    if dag_file.exists():
        return MemoryType.DAG

    raise ValueError("Could not infer memory type from files")
