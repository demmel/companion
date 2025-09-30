#!/usr/bin/env python3
"""
DAG Memory Analysis Script

Analyzes action logs from the DAG memory system to understand how context
and memory evolve over time. Tracks memory lifetimes, retrieval patterns,
context thrashing, and other performance metrics.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import statistics
import csv

# Add the agent_server/src directory to the path for imports
import sys

sys.path.append(str(Path(__file__).parent / "src"))

from agent.memory_dag.action_log import MemoryActionLog
from agent.memory_dag.actions import (
    CheckpointAction,
    AddToContextAction,
    RemoveFromContextAction,
    ApplyTokenDecayAction,
    AddMemoryAction,
    AddContainerAction,
)
from agent.chain_of_action.trigger_history import TriggerHistory


@dataclass
class MemoryLifetime:
    """Tracks a memory's lifetime in context"""

    memory_id: str
    add_turn: int
    remove_turn: Optional[int] = None
    turns_in_context: Optional[int] = None
    initial_tokens: int = 0
    final_tokens: int = 0
    reinforcements: int = 0


@dataclass
class ContextEvent:
    """Represents a context change event"""

    timestamp: datetime
    action_type: str
    memory_id: str
    tokens: int = 0
    turn_number: int = 0


@dataclass
class ThrashAnalysis:
    """Analysis of memory thrashing patterns"""

    memory_id: str
    cycle_count: int = 0  # Number of add/remove cycles
    total_add_events: int = 0
    total_remove_events: int = 0
    average_duration_in_context: float = 0.0
    wasted_tokens: int = 0  # Tokens from memories that were quickly removed


@dataclass
class TurnAnalysis:
    """Analysis of a single conversation turn"""

    turn_number: int
    start_checkpoint: str
    end_checkpoint: str
    start_time: datetime
    end_time: datetime
    memories_added: Set[str] = field(default_factory=set)
    memories_removed: Set[str] = field(default_factory=set)
    containers_added: int = 0
    context_size_start: int = 0
    context_size_end: int = 0
    token_changes: int = 0


class DAGMemoryAnalyzer:
    """Analyzes DAG memory system performance and dynamics"""

    def __init__(self, action_log: MemoryActionLog, trigger_history: TriggerHistory):
        self.action_log = action_log
        self.trigger_history = trigger_history

        # Analysis data structures
        self.memory_lifetimes: Dict[str, List[MemoryLifetime]] = defaultdict(list)
        self.context_events: List[ContextEvent] = []
        self.thrash_analysis: Dict[str, ThrashAnalysis] = {}
        self.turn_analyses: List[TurnAnalysis] = []
        self.current_context_state: Dict[str, int] = {}  # memory_id -> tokens

        # Statistics
        self.retrieval_ages: List[float] = []
        self.retrieval_node_counts: List[int] = []
        self.context_sizes_over_time: List[Tuple[datetime, int]] = []

    @classmethod
    def load_action_log(cls, action_log_path: str) -> "DAGMemoryAnalyzer":
        """Load the action log and trigger history from conversation data"""
        from agent.conversation_persistence import ConversationPersistence

        _action_log_path = Path(action_log_path)
        conversation_dir = _action_log_path.parent
        file_prefix = _action_log_path.stem  # Remove .json extension

        print(f"Loading conversation data from {conversation_dir}/{file_prefix}")

        persistence = ConversationPersistence(str(conversation_dir))
        agent_data = persistence.load_agent_data(file_prefix)
        action_log = agent_data.dag_memory_manager.action_log
        trigger_history = agent_data.trigger_history

        print(f"Loaded {len(action_log.actions)} actions")
        print(f"Loaded trigger history with {len(trigger_history.entries)} entries")

        return cls(action_log, trigger_history)

    def identify_turn_boundaries(self) -> List[Tuple[int, CheckpointAction]]:
        """Identify turn boundaries using checkpoints after pruning"""
        boundaries = []

        for i, action in enumerate(self.action_log.actions):
            if isinstance(action, CheckpointAction):
                # Look for checkpoints that come after pruning operations
                # This is indicated by "prune" in the label or description, or
                # checkpoints that follow remove_from_context actions
                prev_actions = self.action_log.actions[max(0, i - 10) : i]
                has_recent_removal = any(
                    isinstance(a, RemoveFromContextAction) for a in prev_actions
                )

                is_pruning_checkpoint = "context_pruned" in action.label

                if is_pruning_checkpoint:
                    boundaries.append((i, action))

        print(f"Identified {len(boundaries)} turn boundaries")
        return boundaries

    def analyze_memory_lifetimes(self) -> None:
        """Analyze how many turns memories spend in context"""
        print("Analyzing memory lifetimes...")

        current_lifetimes: Dict[str, MemoryLifetime] = {}
        turn_number = 0

        for action in self.action_log.actions:
            if isinstance(action, CheckpointAction):
                # Check if this is a pruning checkpoint (turn boundary)
                if "context_pruned" in action.label:
                    turn_number += 1

            elif isinstance(action, AddToContextAction):
                # Start tracking this memory's lifetime
                lifetime = MemoryLifetime(
                    memory_id=action.memory_id,
                    add_turn=turn_number,
                    initial_tokens=action.initial_tokens,
                )
                current_lifetimes[action.memory_id] = lifetime
                self.current_context_state[action.memory_id] = action.initial_tokens

                # Record context event
                event = ContextEvent(
                    timestamp=action.timestamp,
                    action_type="add_to_context",
                    memory_id=action.memory_id,
                    tokens=action.initial_tokens,
                    turn_number=turn_number,
                )
                self.context_events.append(event)

            elif isinstance(action, RemoveFromContextAction):
                # End tracking for removed memories
                for memory_id in action.memory_ids:
                    if memory_id in current_lifetimes:
                        lifetime = current_lifetimes[memory_id]
                        lifetime.remove_turn = turn_number
                        lifetime.turns_in_context = turn_number - lifetime.add_turn
                        lifetime.final_tokens = self.current_context_state.get(
                            memory_id, 0
                        )
                        self.memory_lifetimes[memory_id].append(lifetime)
                        del current_lifetimes[memory_id]

                    if memory_id in self.current_context_state:
                        final_tokens = self.current_context_state[memory_id]
                        del self.current_context_state[memory_id]
                    else:
                        final_tokens = 0

                    # Record context event
                    event = ContextEvent(
                        timestamp=action.timestamp,
                        action_type="remove_from_context",
                        memory_id=memory_id,
                        tokens=final_tokens,
                        turn_number=turn_number,
                    )
                    self.context_events.append(event)

            elif isinstance(action, ApplyTokenDecayAction):
                # Update token counts for current memories
                for memory_id in self.current_context_state:
                    self.current_context_state[memory_id] = max(
                        0, self.current_context_state[memory_id] - action.decay_amount
                    )
                    if memory_id in current_lifetimes:
                        current_lifetimes[memory_id].reinforcements += 1

            # Track context size over time (number of memories, not tokens)
            context_size = len(self.current_context_state)
            self.context_sizes_over_time.append((action.timestamp, context_size))

    def analyze_thrashing(self) -> None:
        """Analyze context thrashing patterns"""
        print("Analyzing context thrashing...")

        memory_cycles: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)

        # Collect all add/remove events for each memory
        for event in self.context_events:
            memory_cycles[event.memory_id].append((event.action_type, event.timestamp))

        # Analyze thrashing for each memory
        for memory_id, events in memory_cycles.items():
            # Sort events by timestamp
            events.sort(key=lambda x: x[1])

            add_count = sum(
                1 for action_type, _ in events if action_type == "add_to_context"
            )
            remove_count = sum(
                1 for action_type, _ in events if action_type == "remove_from_context"
            )

            # Count cycles (add followed by remove)
            cycles = 0
            durations = []
            wasted_tokens = 0

            i = 0
            while i < len(events) - 1:
                if (
                    events[i][0] == "add_to_context"
                    and events[i + 1][0] == "remove_from_context"
                ):
                    cycles += 1
                    duration = (events[i + 1][1] - events[i][1]).total_seconds()
                    durations.append(duration)

                    # If removed quickly (within 5 minutes), consider tokens wasted
                    if duration < 300:  # 5 minutes
                        # Estimate tokens (would need to track actual tokens)
                        wasted_tokens += 50  # Rough estimate
                    i += 2
                else:
                    i += 1

            avg_duration = statistics.mean(durations) if durations else 0

            analysis = ThrashAnalysis(
                memory_id=memory_id,
                cycle_count=cycles,
                total_add_events=add_count,
                total_remove_events=remove_count,
                average_duration_in_context=avg_duration,
                wasted_tokens=wasted_tokens,
            )

            self.thrash_analysis[memory_id] = analysis

    def analyze_retrieval_patterns(self) -> None:
        """Analyze memory retrieval patterns"""
        print("Analyzing retrieval patterns...")

        # Track when memories are created vs when they're retrieved
        memory_creation_turns: Dict[str, int] = {}
        turn_number = 0
        in_retrieval_phase = False  # Flag to identify pre-pruning retrievals

        for action in self.action_log.actions:
            if isinstance(action, CheckpointAction):
                if "trigger_start" in action.label:
                    # Start of new turn - retrieval phase begins
                    in_retrieval_phase = True
                elif "context_pruned" in action.label:
                    # Pruning checkpoint - retrieval phase ends, turn completes
                    in_retrieval_phase = False
                    turn_number += 1

            elif isinstance(action, AddMemoryAction):
                memory_creation_turns[action.memory.id] = turn_number

            elif isinstance(action, AddToContextAction):
                memory_id = action.memory_id

                # Only count as retrieval if it happens during retrieval phase (before pruning)
                if in_retrieval_phase and memory_id in memory_creation_turns:
                    memory_creation_turn = memory_creation_turns[memory_id]
                    age_in_turns = turn_number - memory_creation_turn

                    # Only count if it's retrieving an older memory (age > 0)
                    if age_in_turns > 0:
                        self.retrieval_ages.append(age_in_turns)

                        # Debug: log first few retrieval examples
                        if len(self.retrieval_ages) <= 5:
                            print(
                                f"Retrieved memory {memory_id[:8]} created turn {memory_creation_turn}, retrieved turn {turn_number}, age {age_in_turns}"
                            )

            elif isinstance(action, AddContainerAction):
                # Container addition - track node counts
                retrieved_memories = action.element_ids
                self.retrieval_node_counts.append(len(retrieved_memories))

        print(
            f"Analyzed {len(self.retrieval_ages)} retrieved memories, {len(self.retrieval_node_counts)} container additions"
        )
        if self.retrieval_ages:
            print(f"Sample retrieval ages: {self.retrieval_ages[:10]}")
            print(
                f"Max turn with memory creation: {max(memory_creation_turns.values()) if memory_creation_turns else 'None'}"
            )
            print(f"Max turn reached: {turn_number}")

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze how patterns change between early and late turns"""
        print("Analyzing temporal patterns...")

        if not self.context_events:
            return {}

        # Find total turns
        max_turn = max(event.turn_number for event in self.context_events)
        if max_turn <= 6:  # Need enough turns to split meaningfully
            return {"note": "Not enough turns for temporal analysis"}

        # Split into early, middle, and late periods
        early_cutoff = max_turn // 3
        late_cutoff = max_turn * 2 // 3

        # Categorize events by time period
        early_events = [e for e in self.context_events if e.turn_number <= early_cutoff]
        middle_events = [
            e
            for e in self.context_events
            if early_cutoff < e.turn_number <= late_cutoff
        ]
        late_events = [e for e in self.context_events if e.turn_number > late_cutoff]

        # Analyze add/remove patterns by period
        def analyze_period_events(events, period_name):
            adds = [e for e in events if e.action_type == "add_to_context"]
            removes = [e for e in events if e.action_type == "remove_from_context"]

            # Count unique memories being added/removed
            unique_adds = len(set(e.memory_id for e in adds))
            unique_removes = len(set(e.memory_id for e in removes))

            # Calculate churn rate (removes/adds)
            churn_rate = unique_removes / unique_adds if unique_adds > 0 else 0

            return {
                f"{period_name}_add_events": len(adds),
                f"{period_name}_remove_events": len(removes),
                f"{period_name}_unique_memories_added": unique_adds,
                f"{period_name}_unique_memories_removed": unique_removes,
                f"{period_name}_churn_rate": churn_rate,
                f"{period_name}_turns": len(set(e.turn_number for e in events)),
            }

        patterns = {}
        patterns.update(analyze_period_events(early_events, "early"))
        patterns.update(analyze_period_events(middle_events, "middle"))
        patterns.update(analyze_period_events(late_events, "late"))

        # Context size evolution patterns
        if self.context_sizes_over_time:
            early_sizes = []
            late_sizes = []

            # We need to map timestamps to turns, so let's use a rough approximation
            # by dividing the timeline into thirds
            total_timeline = len(self.context_sizes_over_time)
            early_timeline = self.context_sizes_over_time[: total_timeline // 3]
            late_timeline = self.context_sizes_over_time[total_timeline * 2 // 3 :]

            early_sizes = [size for _, size in early_timeline]
            late_sizes = [size for _, size in late_timeline]

            if early_sizes and late_sizes:
                patterns.update(
                    {
                        "early_avg_context_size": statistics.mean(early_sizes),
                        "late_avg_context_size": statistics.mean(late_sizes),
                        "context_size_growth": statistics.mean(late_sizes)
                        - statistics.mean(early_sizes),
                    }
                )

        # Memory lifetime patterns by period
        early_lifetimes = []
        late_lifetimes = []

        for lifetimes in self.memory_lifetimes.values():
            for lifetime in lifetimes:
                if lifetime.turns_in_context is not None:
                    if lifetime.add_turn <= early_cutoff:
                        early_lifetimes.append(lifetime.turns_in_context)
                    elif lifetime.add_turn > late_cutoff:
                        late_lifetimes.append(lifetime.turns_in_context)

        if early_lifetimes and late_lifetimes:
            patterns.update(
                {
                    "early_avg_lifetime_turns": statistics.mean(early_lifetimes),
                    "late_avg_lifetime_turns": statistics.mean(late_lifetimes),
                    "lifetime_change": statistics.mean(late_lifetimes)
                    - statistics.mean(early_lifetimes),
                }
            )

        patterns["analysis_periods"] = {
            "early_turns": f"1-{early_cutoff}",
            "middle_turns": f"{early_cutoff+1}-{late_cutoff}",
            "late_turns": f"{late_cutoff+1}-{max_turn}",
            "total_turns": max_turn,
        }

        return patterns

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        print("Generating summary statistics...")

        stats = {}

        # Memory lifetime stats (in turns)
        all_turn_durations = []
        for memory_lifetimes in self.memory_lifetimes.values():
            for lifetime in memory_lifetimes:
                if lifetime.turns_in_context is not None:
                    all_turn_durations.append(lifetime.turns_in_context)

        if all_turn_durations:
            stats["memory_lifetime_stats"] = {
                "mean_turns_in_context": statistics.mean(all_turn_durations),
                "median_turns_in_context": statistics.median(all_turn_durations),
                "min_turns_in_context": min(all_turn_durations),
                "max_turns_in_context": max(all_turn_durations),
                "std_turns_in_context": (
                    statistics.stdev(all_turn_durations)
                    if len(all_turn_durations) > 1
                    else 0
                ),
            }

        # Thrashing stats
        high_thrash_memories = [
            analysis
            for analysis in self.thrash_analysis.values()
            if analysis.cycle_count >= 3
        ]

        stats["thrashing_stats"] = {
            "total_memories_analyzed": len(self.thrash_analysis),
            "high_thrash_memories": len(high_thrash_memories),
            "total_wasted_tokens": sum(
                a.wasted_tokens for a in self.thrash_analysis.values()
            ),
            "avg_cycles_per_memory": (
                statistics.mean([a.cycle_count for a in self.thrash_analysis.values()])
                if self.thrash_analysis
                else 0
            ),
        }

        # Retrieval stats (age in turns relative to most recent memory)
        if self.retrieval_ages:
            stats["retrieval_stats"] = {
                "mean_age_turns": statistics.mean(self.retrieval_ages),
                "median_age_turns": statistics.median(self.retrieval_ages),
                "min_age_turns": min(self.retrieval_ages),
                "max_age_turns": max(self.retrieval_ages),
            }

        if self.retrieval_node_counts:
            stats["retrieval_node_stats"] = {
                "mean_nodes_per_retrieval": statistics.mean(self.retrieval_node_counts),
                "median_nodes_per_retrieval": statistics.median(
                    self.retrieval_node_counts
                ),
                "max_nodes_per_retrieval": max(self.retrieval_node_counts),
            }

        # Context evolution stats (number of memories in context)
        if self.context_sizes_over_time:
            sizes = [size for _, size in self.context_sizes_over_time]
            stats["context_evolution_stats"] = {
                "mean_memories_in_context": statistics.mean(sizes),
                "max_memories_in_context": max(sizes),
                "min_memories_in_context": min(sizes),
                "final_memories_in_context": sizes[-1],
            }

        return stats

    def save_detailed_data(self, output_dir: str) -> None:
        """Save detailed analysis data to CSV files"""
        print(f"Saving detailed data to {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Memory lifetimes
        with open(output_path / "memory_lifetimes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "memory_id",
                    "add_turn",
                    "remove_turn",
                    "turns_in_context",
                    "initial_tokens",
                    "final_tokens",
                    "reinforcements",
                ]
            )

            for memory_id, lifetimes in self.memory_lifetimes.items():
                for lifetime in lifetimes:
                    writer.writerow(
                        [
                            memory_id,
                            lifetime.add_turn,
                            lifetime.remove_turn,
                            lifetime.turns_in_context,
                            lifetime.initial_tokens,
                            lifetime.final_tokens,
                            lifetime.reinforcements,
                        ]
                    )

        # Thrashing analysis
        with open(output_path / "thrash_analysis.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "memory_id",
                    "cycle_count",
                    "add_events",
                    "remove_events",
                    "avg_duration_seconds",
                    "wasted_tokens",
                ]
            )

            for memory_id, analysis in self.thrash_analysis.items():
                writer.writerow(
                    [
                        memory_id,
                        analysis.cycle_count,
                        analysis.total_add_events,
                        analysis.total_remove_events,
                        analysis.average_duration_in_context,
                        analysis.wasted_tokens,
                    ]
                )

        # Context events
        with open(output_path / "context_events.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "action_type", "memory_id", "tokens", "turn_number"]
            )

            for event in self.context_events:
                writer.writerow(
                    [
                        event.timestamp,
                        event.action_type,
                        event.memory_id,
                        event.tokens,
                        event.turn_number,
                    ]
                )

        # Context size over time
        with open(output_path / "context_size_timeline.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "memories_in_context"])

            for timestamp, size in self.context_sizes_over_time:
                writer.writerow([timestamp, size])

    def run_full_analysis(
        self, output_dir: str = "dag_analysis_output"
    ) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        print("Starting DAG Memory Analysis")
        print("=" * 50)

        self.analyze_memory_lifetimes()
        self.analyze_thrashing()
        self.analyze_retrieval_patterns()

        stats = self.generate_summary_stats()
        temporal_patterns = self.analyze_temporal_patterns()
        stats["temporal_patterns"] = temporal_patterns

        self.save_detailed_data(output_dir)

        return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze DAG memory system performance"
    )
    parser.add_argument("action_log", help="Path to the action log JSON file")
    parser.add_argument(
        "-o",
        "--output",
        default="dag_analysis_output",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    analyzer = DAGMemoryAnalyzer.load_action_log(args.action_log)
    stats = analyzer.run_full_analysis(args.output)

    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    for category, category_stats in stats.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for key, value in category_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nDetailed data saved to: {args.output}/")
    print("Files generated:")
    print("  - memory_lifetimes.csv")
    print("  - thrash_analysis.csv")
    print("  - context_events.csv")
    print("  - context_size_timeline.csv")


if __name__ == "__main__":
    main()
