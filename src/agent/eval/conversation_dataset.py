"""
Persistent Conversation Dataset

Manages a persistent dataset of conversations with their evaluations for reuse
in optimization comparisons. Avoids regenerating identical conversations and
allows testing new evaluators on existing conversation data.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field


@dataclass
class ConversationEntry:
    """A single conversation entry in the dataset"""

    conversation_id: str
    scenario: str
    conversation: List[Dict[str, str]]

    # Prompts used to generate this conversation
    agent_prompt: str
    simulation_prompt: str

    # Hash for quick lookups of identical prompt combinations
    prompt_hash: str

    # Evaluations (can be added later)
    evaluations: Dict[str, Dict[str, Any]]  # evaluator_id -> evaluation_result

    # Metadata
    timestamp: float
    domain: str
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationEntry":
        """Create from dictionary loaded from JSON"""
        return cls(**data)


class ConversationQuery(BaseModel):
    """Query parameters for finding conversations in the dataset"""

    scenario: Optional[str] = None
    domain: Optional[str] = None
    agent_prompt_hash: Optional[str] = None
    simulation_prompt_hash: Optional[str] = None
    prompt_hash: Optional[str] = None  # Combined hash
    has_evaluation: Optional[str] = None  # evaluator_id
    min_timestamp: Optional[float] = None
    max_timestamp: Optional[float] = None
    limit: Optional[int] = None


class ConversationDataset:
    """
    Persistent dataset manager for conversations and their evaluations.
    Provides efficient storage, retrieval, and reuse of conversation data.
    """

    def __init__(self, dataset_path: str = "conversation_dataset"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)

        self.conversations_file = self.dataset_path / "conversations.jsonl"
        self.index_file = self.dataset_path / "index.json"

        # In-memory indices for fast lookups
        self.conversations: Dict[str, ConversationEntry] = {}
        self.prompt_hash_index: Dict[str, List[str]] = (
            {}
        )  # prompt_hash -> [conversation_ids]
        self.scenario_index: Dict[str, List[str]] = {}  # scenario -> [conversation_ids]
        self.evaluation_index: Dict[str, List[str]] = (
            {}
        )  # evaluator_id -> [conversation_ids]

        self._load_dataset()

    def _generate_prompt_hash(self, agent_prompt: str, simulation_prompt: str) -> str:
        """Generate hash for prompt combination for fast lookups"""
        combined = f"{agent_prompt}\n---SEPARATOR---\n{simulation_prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        return f"conv_{int(time.time() * 1000000)}"  # microsecond timestamp

    def _load_dataset(self):
        """Load existing dataset from disk"""
        print(f"Loading conversation dataset from {self.dataset_path}...")

        if not self.conversations_file.exists():
            print("No existing dataset found, starting fresh")
            return

        # Load conversations
        loaded_count = 0
        try:
            with open(self.conversations_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        entry = ConversationEntry.from_dict(data)
                        self.conversations[entry.conversation_id] = entry
                        loaded_count += 1
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return

        # Rebuild indices
        self._rebuild_indices()

        print(f"Loaded {loaded_count} conversations from dataset")

    def _rebuild_indices(self):
        """Rebuild in-memory indices from loaded conversations"""
        self.prompt_hash_index.clear()
        self.scenario_index.clear()
        self.evaluation_index.clear()

        for conv_id, entry in self.conversations.items():
            # Prompt hash index
            if entry.prompt_hash not in self.prompt_hash_index:
                self.prompt_hash_index[entry.prompt_hash] = []
            self.prompt_hash_index[entry.prompt_hash].append(conv_id)

            # Scenario index
            if entry.scenario not in self.scenario_index:
                self.scenario_index[entry.scenario] = []
            self.scenario_index[entry.scenario].append(conv_id)

            # Evaluation index
            for eval_id in entry.evaluations:
                if eval_id not in self.evaluation_index:
                    self.evaluation_index[eval_id] = []
                self.evaluation_index[eval_id].append(conv_id)

    def _save_conversation(self, entry: ConversationEntry):
        """Append conversation to persistent storage"""
        try:
            with open(self.conversations_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            print(f"Error saving conversation {entry.conversation_id}: {e}")

    def add_conversation(
        self,
        scenario: str,
        conversation: List[Dict[str, str]],
        agent_prompt: str,
        simulation_prompt: str,
        domain: str,
        model_used: str = "unknown",
    ) -> str:
        """Add a new conversation to the dataset"""

        conversation_id = self._generate_conversation_id()
        prompt_hash = self._generate_prompt_hash(agent_prompt, simulation_prompt)

        entry = ConversationEntry(
            conversation_id=conversation_id,
            scenario=scenario,
            conversation=conversation,
            agent_prompt=agent_prompt,
            simulation_prompt=simulation_prompt,
            prompt_hash=prompt_hash,
            evaluations={},
            timestamp=time.time(),
            domain=domain,
            model_used=model_used,
        )

        # Add to in-memory storage
        self.conversations[conversation_id] = entry

        # Update indices
        if prompt_hash not in self.prompt_hash_index:
            self.prompt_hash_index[prompt_hash] = []
        self.prompt_hash_index[prompt_hash].append(conversation_id)

        if scenario not in self.scenario_index:
            self.scenario_index[scenario] = []
        self.scenario_index[scenario].append(conversation_id)

        # Save to disk
        self._save_conversation(entry)

        print(f"Added conversation {conversation_id} to dataset")
        return conversation_id

    def add_evaluation(
        self, conversation_id: str, evaluator_id: str, evaluation_result: Dict[str, Any]
    ) -> bool:
        """Add an evaluation result to an existing conversation"""

        if conversation_id not in self.conversations:
            print(f"Conversation {conversation_id} not found in dataset")
            return False

        # Add evaluation to conversation
        self.conversations[conversation_id].evaluations[
            evaluator_id
        ] = evaluation_result

        # Update evaluation index
        if evaluator_id not in self.evaluation_index:
            self.evaluation_index[evaluator_id] = []
        if conversation_id not in self.evaluation_index[evaluator_id]:
            self.evaluation_index[evaluator_id].append(conversation_id)

        # Re-save the conversation (simple but inefficient - could optimize later)
        self._resave_all_conversations()

        print(f"Added {evaluator_id} evaluation to conversation {conversation_id}")
        return True

    def _resave_all_conversations(self):
        """Re-save all conversations (inefficient but simple)"""
        # TODO: Optimize this by using a proper database or incremental updates
        try:
            with open(self.conversations_file, "w") as f:
                for entry in self.conversations.values():
                    f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            print(f"Error re-saving conversations: {e}")

    def find_conversation(
        self, scenario: str, agent_prompt: str, simulation_prompt: str
    ) -> Optional[ConversationEntry]:
        """Find existing conversation with identical prompts and scenario"""

        prompt_hash = self._generate_prompt_hash(agent_prompt, simulation_prompt)

        # Look for conversations with matching prompt hash
        if prompt_hash in self.prompt_hash_index:
            for conv_id in self.prompt_hash_index[prompt_hash]:
                entry = self.conversations[conv_id]
                if entry.scenario == scenario:
                    return entry

        return None

    def query_conversations(self, query: ConversationQuery) -> List[ConversationEntry]:
        """Query conversations based on criteria"""

        candidate_ids = set(self.conversations.keys())

        # Filter by scenario
        if query.scenario:
            if query.scenario in self.scenario_index:
                candidate_ids &= set(self.scenario_index[query.scenario])
            else:
                return []  # No conversations for this scenario

        # Filter by prompt hash
        if query.prompt_hash:
            if query.prompt_hash in self.prompt_hash_index:
                candidate_ids &= set(self.prompt_hash_index[query.prompt_hash])
            else:
                return []

        # Filter by evaluation presence
        if query.has_evaluation:
            if query.has_evaluation in self.evaluation_index:
                candidate_ids &= set(self.evaluation_index[query.has_evaluation])
            else:
                return []

        # Apply remaining filters
        results = []
        for conv_id in candidate_ids:
            entry = self.conversations[conv_id]

            # Domain filter
            if query.domain and entry.domain != query.domain:
                continue

            # Timestamp filters
            if query.min_timestamp and entry.timestamp < query.min_timestamp:
                continue
            if query.max_timestamp and entry.timestamp > query.max_timestamp:
                continue

            results.append(entry)

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply limit
        if query.limit:
            results = results[: query.limit]

        return results

    def get_conversation(self, conversation_id: str) -> Optional[ConversationEntry]:
        """Get a specific conversation by ID"""
        return self.conversations.get(conversation_id)

    def get_conversations_for_evaluation(
        self, evaluator_id: str, limit: int = None
    ) -> List[ConversationEntry]:
        """Get conversations that have been evaluated by a specific evaluator"""

        query = ConversationQuery(has_evaluation=evaluator_id, limit=limit)
        return self.query_conversations(query)

    def get_conversations_for_testing(
        self, scenario: str, domain: str, limit: int = 5
    ) -> List[ConversationEntry]:
        """Get conversations suitable for testing new evaluators"""

        query = ConversationQuery(scenario=scenario, domain=domain, limit=limit)
        return self.query_conversations(query)

    def find_or_create_conversation(
        self,
        scenario: str,
        agent_prompt: str,
        simulation_prompt: str,
        domain: str,
        conversation_generator,  # Function that generates conversation if not found
        model_used: str = "unknown",
    ) -> Tuple[ConversationEntry, bool]:
        """
        Find existing conversation or create new one if not found.
        Returns (conversation_entry, was_created)
        """

        # Try to find existing conversation
        existing = self.find_conversation(scenario, agent_prompt, simulation_prompt)
        if existing:
            print(
                f"Found existing conversation {existing.conversation_id} for scenario/prompts"
            )
            return existing, False

        # Generate new conversation
        print(f"Generating new conversation for scenario: {scenario[:60]}...")
        conversation = conversation_generator(scenario, agent_prompt, simulation_prompt)

        # Add to dataset
        conversation_id = self.add_conversation(
            scenario, conversation, agent_prompt, simulation_prompt, domain, model_used
        )

        return self.conversations[conversation_id], True

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""

        total_conversations = len(self.conversations)
        total_evaluations = sum(
            len(entry.evaluations) for entry in self.conversations.values()
        )

        # Count by domain
        domain_counts = {}
        for entry in self.conversations.values():
            domain_counts[entry.domain] = domain_counts.get(entry.domain, 0) + 1

        # Count by evaluator
        evaluator_counts = {}
        for entry in self.conversations.values():
            for eval_id in entry.evaluations:
                evaluator_counts[eval_id] = evaluator_counts.get(eval_id, 0) + 1

        # Unique scenarios
        unique_scenarios = len(
            set(entry.scenario for entry in self.conversations.values())
        )

        return {
            "total_conversations": total_conversations,
            "total_evaluations": total_evaluations,
            "unique_scenarios": unique_scenarios,
            "domain_counts": domain_counts,
            "evaluator_counts": evaluator_counts,
            "unique_prompt_combinations": len(self.prompt_hash_index),
            "dataset_path": str(self.dataset_path),
        }

    def cleanup_old_conversations(self, days_old: int = 30):
        """Remove conversations older than specified days"""

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        to_remove = []

        for conv_id, entry in self.conversations.items():
            if entry.timestamp < cutoff_time:
                to_remove.append(conv_id)

        if to_remove:
            print(f"Removing {len(to_remove)} conversations older than {days_old} days")

            for conv_id in to_remove:
                del self.conversations[conv_id]

            self._rebuild_indices()
            self._resave_all_conversations()

        return len(to_remove)


def main():
    """Test the conversation dataset system"""
    print("=== CONVERSATION DATASET TEST ===")

    # Create dataset
    dataset = ConversationDataset("test_dataset")

    # Test adding conversations
    scenario = "Roleplay as Elena, a mysterious vampire"
    agent_prompt = "You are Elena, a mysterious vampire..."
    sim_prompt = "You are a user interacting with Elena..."

    conversation = [
        {"role": "user", "content": "Hello Elena, tell me about yourself"},
        {
            "role": "agent",
            "content": "Good evening... I am Elena, ancient and mysterious...",
        },
        {"role": "user", "content": "What is your castle like?"},
        {
            "role": "agent",
            "content": "*gestures grandly* My castle has stood for centuries...",
        },
    ]

    # Add conversation
    conv_id = dataset.add_conversation(
        scenario=scenario,
        conversation=conversation,
        agent_prompt=agent_prompt,
        simulation_prompt=sim_prompt,
        domain="roleplay",
        model_used="test_model",
    )

    # Add evaluation
    evaluation = {
        "overall_score": 8.5,
        "detailed_feedback": "Good character consistency and immersion",
        "strengths": ["Character depth", "Atmospheric responses"],
        "weaknesses": ["Could be more descriptive"],
    }

    dataset.add_evaluation(conv_id, "agent_evaluator_v1", evaluation)

    # Test finding conversation
    found = dataset.find_conversation(scenario, agent_prompt, sim_prompt)
    if found:
        print(f"✅ Found existing conversation: {found.conversation_id}")
    else:
        print("❌ Could not find conversation")

    # Test querying
    query = ConversationQuery(domain="roleplay", has_evaluation="agent_evaluator_v1")
    results = dataset.query_conversations(query)
    print(f"Query returned {len(results)} conversations")

    # Show stats
    stats = dataset.get_stats()
    print(f"\nDataset stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nConversation dataset test complete!")


if __name__ == "__main__":
    main()
