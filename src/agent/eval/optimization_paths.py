"""
Optimization Path Manager

Centralized path management for the optimization framework, similar to how
the agent has organized path management.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class OptimizationPaths:
    """Container for all optimization-related paths"""

    # Root directories
    root_dir: Path
    preferences_dir: Path
    history_dir: Path
    meta_history_dir: Path
    test_conversations_dir: Path
    optimized_prompts_dir: Path

    # Preference files
    semantic_preferences_file: Path

    # History files
    optimization_history_file: Path
    meta_optimization_history_file: Path

    # Strategy files
    best_strategy_file: Path

    # Test conversation files
    test_conversations_file: Path
    evaluation_benchmarks_file: Path


class OptimizationPathManager:
    """Manages all file paths and directories for the optimization system"""

    def __init__(self, base_dir: str = "optimization_data", domain: str = "roleplay"):
        self.base_dir = Path(base_dir)
        self.domain = domain
        self.paths = self._setup_paths()
        self._ensure_directories()

    def _setup_paths(self) -> OptimizationPaths:
        """Set up all optimization paths"""
        root = self.base_dir

        return OptimizationPaths(
            # Root directories
            root_dir=root,
            preferences_dir=root / "preferences",
            history_dir=root / "history",
            meta_history_dir=root / "meta_history",
            test_conversations_dir=root / "test_conversations",
            optimized_prompts_dir=root / "optimized_prompts",
            # Preference files
            semantic_preferences_file=root
            / "preferences"
            / "semantic_preferences_v2.json",
            # History files
            optimization_history_file=root
            / "history"
            / f"{self.domain}_optimization_history.json",
            meta_optimization_history_file=root
            / "meta_history"
            / "meta_optimization_history.json",
            # Strategy files
            best_strategy_file=root
            / "meta_history"
            / "best_optimization_strategy.json",
            # Test conversation files
            test_conversations_file=root
            / "test_conversations"
            / f"{self.domain}_test_conversations.json",
            evaluation_benchmarks_file=root
            / "test_conversations"
            / f"{self.domain}_evaluation_benchmarks.json",
        )

    def _ensure_directories(self):
        """Create all necessary directories"""
        directories = [
            self.paths.root_dir,
            self.paths.preferences_dir,
            self.paths.history_dir,
            self.paths.meta_history_dir,
            self.paths.test_conversations_dir,
            self.paths.optimized_prompts_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # Optimized prompt file methods
    def get_optimized_prompt_file(self, prompt_type: str) -> Path:
        """Get path for optimized prompt file"""
        return (
            self.paths.optimized_prompts_dir
            / f"{self.domain}_{prompt_type}_optimized.txt"
        )

    def save_optimized_prompt(self, prompt_type: str, prompt: str):
        """Save optimized prompt to file"""
        prompt_file = self.get_optimized_prompt_file(prompt_type)
        with open(prompt_file, "w") as f:
            f.write(prompt)
        return prompt_file

    def load_optimized_prompt(self, prompt_type: str) -> Optional[str]:
        """Load optimized prompt from file"""
        prompt_file = self.get_optimized_prompt_file(prompt_type)
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                return f.read()
        return None

    # Test conversation methods
    def save_test_conversations(self, conversations: List[Dict[str, Any]]):
        """Save test conversations for evaluation prompt testing"""
        data = {
            "domain": self.domain,
            "timestamp": time.time(),
            "conversations": conversations,
        }

        with open(self.paths.test_conversations_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_test_conversations(self) -> List[Dict[str, Any]]:
        """Load test conversations"""
        if self.paths.test_conversations_file.exists():
            try:
                with open(self.paths.test_conversations_file, "r") as f:
                    data = json.load(f)
                return data.get("conversations", [])
            except Exception:
                return []
        return []

    def save_evaluation_benchmarks(self, benchmarks: List[Dict[str, Any]]):
        """Save evaluation benchmarks for testing evaluation prompts"""
        data = {
            "domain": self.domain,
            "timestamp": time.time(),
            "benchmarks": benchmarks,
        }

        with open(self.paths.evaluation_benchmarks_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_evaluation_benchmarks(self) -> List[Dict[str, Any]]:
        """Load evaluation benchmarks"""
        if self.paths.evaluation_benchmarks_file.exists():
            try:
                with open(self.paths.evaluation_benchmarks_file, "r") as f:
                    data = json.load(f)
                return data.get("benchmarks", [])
            except Exception:
                return []
        return []

    # Optimization run archival
    def archive_optimization_run(self, run_result: Dict[str, Any]) -> Path:
        """Archive a completed optimization run"""
        timestamp = int(time.time())
        run_type = run_result.get("prompt_type", "unknown")

        archive_file = self.paths.history_dir / f"{run_type}_run_{timestamp}.json"

        with open(archive_file, "w") as f:
            json.dump(run_result, f, indent=2)

        return archive_file

    def get_archived_runs(
        self, prompt_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get archived optimization runs"""
        runs = []

        pattern = f"{prompt_type}_run_*.json" if prompt_type else "*_run_*.json"

        for run_file in self.paths.history_dir.glob(pattern):
            try:
                with open(run_file, "r") as f:
                    run_data = json.load(f)
                runs.append(run_data)
            except Exception:
                continue

        # Sort by timestamp if available
        runs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return runs

    # Meta-optimization strategy methods
    def save_best_strategy(self, strategy: Dict[str, Any]):
        """Save the best meta-optimization strategy"""
        data = {"domain": self.domain, "timestamp": time.time(), "strategy": strategy}

        with open(self.paths.best_strategy_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_best_strategy(self) -> Optional[Dict[str, Any]]:
        """Load the best meta-optimization strategy"""
        if self.paths.best_strategy_file.exists():
            try:
                with open(self.paths.best_strategy_file, "r") as f:
                    data = json.load(f)
                return data.get("strategy")
            except Exception:
                return None
        return None

    # Utility methods
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old optimization files"""
        import time

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        for directory in [self.paths.history_dir, self.paths.test_conversations_dir]:
            for file_path in directory.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                except Exception:
                    continue

    def get_status(self) -> Dict[str, Any]:
        """Get status of optimization file system"""

        def count_files(directory: Path, pattern: str = "*") -> int:
            if directory.exists():
                return len(list(directory.glob(pattern)))
            return 0

        return {
            "base_directory": str(self.paths.root_dir),
            "domain": self.domain,
            "directories_exist": {
                "preferences": self.paths.preferences_dir.exists(),
                "history": self.paths.history_dir.exists(),
                "meta_history": self.paths.meta_history_dir.exists(),
                "test_conversations": self.paths.test_conversations_dir.exists(),
                "optimized_prompts": self.paths.optimized_prompts_dir.exists(),
            },
            "file_counts": {
                "optimization_runs": count_files(
                    self.paths.history_dir, "*_run_*.json"
                ),
                "optimized_prompts": count_files(
                    self.paths.optimized_prompts_dir, "*.txt"
                ),
                "test_conversations": (
                    1 if self.paths.test_conversations_file.exists() else 0
                ),
                "evaluation_benchmarks": (
                    1 if self.paths.evaluation_benchmarks_file.exists() else 0
                ),
            },
            "key_files_exist": {
                "semantic_preferences": self.paths.semantic_preferences_file.exists(),
                "best_strategy": self.paths.best_strategy_file.exists(),
                "test_conversations": self.paths.test_conversations_file.exists(),
            },
        }

    def __str__(self) -> str:
        """String representation showing key paths"""
        return (
            f"OptimizationPathManager(domain={self.domain}, base={self.paths.root_dir})"
        )


# Convenience functions for backward compatibility
def get_optimization_paths(
    domain: str = "roleplay", base_dir: str = "optimization_data"
) -> OptimizationPathManager:
    """Get optimization path manager for a domain"""
    return OptimizationPathManager(base_dir=base_dir, domain=domain)


def main():
    """Test the optimization path manager"""
    print("=== OPTIMIZATION PATH MANAGER TEST ===")

    # Create path manager
    path_manager = OptimizationPathManager("test_optimization_data", "roleplay")

    print(f"Path manager: {path_manager}")

    # Test saving/loading optimized prompts
    print("\n🧪 Testing optimized prompt management...")
    test_prompt = (
        "You are an expert roleplay assistant with enhanced character consistency."
    )
    saved_file = path_manager.save_optimized_prompt("agent", test_prompt)
    print(f"Saved prompt to: {saved_file}")

    loaded_prompt = path_manager.load_optimized_prompt("agent")
    print(f"Loaded prompt: {loaded_prompt[:50]}...")

    # Test test conversation management
    print("\n🧪 Testing test conversation management...")
    test_conversations = [
        {
            "conversation": [
                {"role": "user", "content": "Test"},
                {"role": "agent", "content": "Response"},
            ],
            "scenario": "Test scenario",
            "quality": "high",
        }
    ]
    path_manager.save_test_conversations(test_conversations)
    loaded_conversations = path_manager.load_test_conversations()
    print(f"Saved and loaded {len(loaded_conversations)} test conversations")

    # Show status
    print("\n📊 Path manager status:")
    status = path_manager.get_status()
    for section, data in status.items():
        if isinstance(data, dict):
            print(f"  {section}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {section}: {data}")

    print("\n✅ Optimization path manager test complete!")


if __name__ == "__main__":
    main()
