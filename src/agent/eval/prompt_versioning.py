"""
Prompt Versioning System

Handles progressive saving of prompts during optimization with diff visualization
and evolution tracking.
"""

import time
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .optimization_paths import OptimizationPathManager
from agent.progress import ProgressReporter


@dataclass
class PromptVersion:
    """Metadata for a prompt version"""

    prompt_type: str
    run_id: str
    cycle: int
    step: str
    timestamp: float
    file_path: Path
    metadata: Dict[str, Any]


class PromptVersionManager:
    """Manages versioned prompt saving and diff generation during optimization"""

    def __init__(
        self, path_manager: OptimizationPathManager, progress: ProgressReporter
    ):
        self.path_manager = path_manager
        self.progress = progress

    def save_prompt_version(
        self,
        prompt_type: str,
        prompt: str,
        run_id: str,
        cycle: int,
        step: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """Save a versioned prompt with metadata"""
        timestamp = time.time()
        timestamp_str = int(timestamp)

        version_file = (
            self.path_manager.paths.optimized_prompts_dir
            / f"{self.path_manager.domain}_{prompt_type}_{run_id}_cycle{cycle:02d}_{step}_{timestamp_str}.txt"
        )

        # Create metadata header
        meta = metadata or {}
        header_lines = [
            "# Prompt Version Metadata",
            f"# Run ID: {run_id}",
            f"# Cycle: {cycle}",
            f"# Step: {step}",
            f"# Timestamp: {timestamp_str}",
            f"# Type: {prompt_type}",
            f"# Domain: {self.path_manager.domain}",
        ]

        for key, value in meta.items():
            header_lines.append(f"# {key}: {value}")

        header_lines.extend(["", "# === PROMPT CONTENT ===", ""])

        with open(version_file, "w") as f:
            f.write("\n".join(header_lines) + prompt)

        return PromptVersion(
            prompt_type=prompt_type,
            run_id=run_id,
            cycle=cycle,
            step=step,
            timestamp=timestamp,
            file_path=version_file,
            metadata=meta,
        )

    def get_prompt_versions(
        self, prompt_type: str, run_id: Optional[str] = None
    ) -> List[PromptVersion]:
        """Get all versions of a prompt type, optionally filtered by run_id"""
        if run_id:
            pattern = f"{self.path_manager.domain}_{prompt_type}_{run_id}_cycle*.txt"
        else:
            pattern = f"{self.path_manager.domain}_{prompt_type}_*_cycle*.txt"

        versions = []
        for version_file in self.path_manager.paths.optimized_prompts_dir.glob(pattern):
            try:
                _, metadata = self._load_prompt_version(version_file)
                version = PromptVersion(
                    prompt_type=prompt_type,
                    run_id=metadata.get("Run ID", "unknown"),
                    cycle=int(metadata.get("Cycle", 0)),
                    step=metadata.get("Step", "unknown"),
                    timestamp=float(metadata.get("Timestamp", 0)),
                    file_path=version_file,
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k
                        not in [
                            "Run ID",
                            "Cycle",
                            "Step",
                            "Timestamp",
                            "Type",
                            "Domain",
                        ]
                    },
                )
                versions.append(version)
            except (ValueError, KeyError):
                continue

        # Sort by cycle, then by timestamp
        return sorted(versions, key=lambda v: (v.cycle, v.timestamp))

    def _load_prompt_version(self, version_file: Path) -> Tuple[str, Dict[str, str]]:
        """Load a versioned prompt and extract its metadata"""
        with open(version_file, "r") as f:
            content = f.read()

        lines = content.split("\n")
        metadata = {}
        prompt_content = []
        in_prompt = False

        for line in lines:
            if line.startswith("# ") and not in_prompt:
                if ": " in line:
                    key, value = line[2:].split(": ", 1)
                    metadata[key] = value
            elif line == "# === PROMPT CONTENT ===":
                in_prompt = True
                continue
            elif in_prompt:
                prompt_content.append(line)

        return "\n".join(prompt_content).strip(), metadata

    def load_prompt_content(self, version: PromptVersion) -> str:
        """Load the prompt content from a version"""
        content, _ = self._load_prompt_version(version.file_path)
        return content

    def create_prompt_diff(
        self,
        old_prompt: str,
        new_prompt: str,
        old_label: str = "Previous",
        new_label: str = "Current",
    ) -> str:
        """Create a human-readable diff between two prompts"""
        old_lines = old_prompt.splitlines(keepends=True)
        new_lines = new_prompt.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                old_lines, new_lines, fromfile=old_label, tofile=new_label, lineterm=""
            )
        )

        if not diff:
            return "No changes detected."

        return "".join(diff)

    def get_compact_diff_summary(self, old_prompt: str, new_prompt: str) -> str:
        """Get a compact summary of what changed between prompts"""
        if old_prompt == new_prompt:
            return "No changes"

        old_lines = old_prompt.splitlines()
        new_lines = new_prompt.splitlines()

        # Count changes
        added_lines = 0
        removed_lines = 0
        modified_lines = 0

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "delete":
                removed_lines += i2 - i1
            elif tag == "insert":
                added_lines += j2 - j1
            elif tag == "replace":
                modified_lines += max(i2 - i1, j2 - j1)

        changes = []
        if added_lines:
            changes.append(f"+{added_lines} lines")
        if removed_lines:
            changes.append(f"-{removed_lines} lines")
        if modified_lines:
            changes.append(f"~{modified_lines} lines")

        return ", ".join(changes) if changes else "Minor changes"

    def display_prompt_diff(
        self, old_prompt: str, new_prompt: str, title: str = "Prompt Changes"
    ):
        """Display a prompt diff in a readable format using progress reporter"""
        self.progress.print(f"\n{'='*60}")
        self.progress.print(f"📝 {title}")
        self.progress.print(f"{'='*60}")

        summary = self.get_compact_diff_summary(old_prompt, new_prompt)
        self.progress.print(f"Summary: {summary}")

        if old_prompt == new_prompt:
            self.progress.print("No changes detected.")
            return

        # Show complete diff with proper formatting
        diff = self.create_prompt_diff(old_prompt, new_prompt)
        diff_lines = diff.split("\n")

        self.progress.print("\nFull diff:")
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---"):
                continue  # Skip file headers
            elif line.startswith("@@"):
                self.progress.print(f"[dim]{line}[/dim]")
            elif line.startswith("+"):
                self.progress.print(f"[green]+ {line[1:]}[/green]")
            elif line.startswith("-"):
                self.progress.print(f"[red]- {line[1:]}[/red]")
            else:
                # Context line
                self.progress.print(f"  {line}")
        
        self.progress.print("")  # Add spacing after diff

    def save_prompt_with_diff(
        self,
        prompt_type: str,
        new_prompt: str,
        old_prompt: str,
        run_id: str,
        cycle: int,
        step: str,
        show_diff: bool = True,
    ) -> PromptVersion:
        """Save a prompt version and optionally display the diff"""

        # Show diff if requested
        if show_diff and old_prompt:
            self.display_prompt_diff(
                old_prompt,
                new_prompt,
                f"{prompt_type.title()} Prompt Changes - Cycle {cycle}, {step}",
            )

        # Save the version
        version = self.save_prompt_version(
            prompt_type=prompt_type,
            prompt=new_prompt,
            run_id=run_id,
            cycle=cycle,
            step=step,
            metadata={
                "previous_prompt_length": len(old_prompt) if old_prompt else 0,
                "new_prompt_length": len(new_prompt),
                "diff_summary": self.get_compact_diff_summary(
                    old_prompt or "", new_prompt
                ),
            },
        )

        self.progress.print(
            f"💾 Saved {prompt_type} prompt version: {version.file_path.name}"
        )

        return version

    def generate_final_summary(self, run_id: str) -> str:
        """Generate a summary of all prompt changes for a run"""
        prompt_types = ["agent", "simulation", "agent_eval", "sim_eval"]

        summary_lines = [
            f"# Optimization Run Summary - {run_id}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Prompt Evolution Summary",
            "",
        ]

        for prompt_type in prompt_types:
            versions = self.get_prompt_versions(prompt_type, run_id)

            if len(versions) <= 1:
                summary_lines.extend(
                    [
                        f"### {prompt_type.title()} Prompt",
                        "No changes during optimization",
                        "",
                    ]
                )
                continue

            summary_lines.extend(
                [
                    f"### {prompt_type.title()} Prompt",
                    f"**Versions:** {len(versions)}",
                    "",
                ]
            )

            # Show evolution summary
            for i, version in enumerate(versions[1:], 1):  # Skip first version
                prev_version = versions[i - 1]
                prev_content = self.load_prompt_content(prev_version)
                curr_content = self.load_prompt_content(version)

                diff_summary = self.get_compact_diff_summary(prev_content, curr_content)
                summary_lines.append(
                    f"- Cycle {version.cycle}, {version.step}: {diff_summary}"
                )

            summary_lines.append("")

        return "\n".join(summary_lines)

    def generate_evolution_report(self, prompt_type: str, run_id: str) -> str:
        """Generate a comprehensive report of prompt evolution for a run"""
        versions = self.get_prompt_versions(prompt_type, run_id)

        if len(versions) < 1:
            return f"No versions found for {prompt_type} in run {run_id}"

        report = [
            f"# Prompt Evolution Report",
            f"**Type:** {prompt_type}",
            f"**Run ID:** {run_id}",
            f"**Total Versions:** {len(versions)}",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Version History",
            "",
        ]

        previous_prompt = None
        for i, version in enumerate(versions):
            prompt_content = self.load_prompt_content(version)

            timestamp_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(version.timestamp)
            )

            report.extend(
                [
                    f"### Version {i+1}: Cycle {version.cycle}, Step {version.step}",
                    f"**File:** {version.file_path.name}",
                    f"**Time:** {timestamp_str}",
                    "",
                ]
            )

            # Add metadata if present
            if version.metadata:
                report.append("**Metadata:**")
                for key, value in version.metadata.items():
                    report.append(f"- {key}: {value}")
                report.append("")

            # Add diff if we have a previous version
            if previous_prompt is not None:
                diff = self.create_prompt_diff(
                    previous_prompt, prompt_content, f"Version {i}", f"Version {i+1}"
                )
                if diff != "No changes detected.":
                    summary = self.get_compact_diff_summary(
                        previous_prompt, prompt_content
                    )
                    report.extend(
                        [
                            f"**Changes from previous version:** {summary}",
                            "",
                            "<details><summary>View detailed diff</summary>",
                            "",
                            "```diff",
                            diff,
                            "```",
                            "",
                            "</details>",
                            "",
                        ]
                    )
                else:
                    report.extend(["**No changes from previous version**", ""])
            else:
                report.extend(["**Initial version**", ""])

            previous_prompt = prompt_content

        return "\n".join(report)

    def save_evolution_report(self, prompt_type: str, run_id: str) -> Path:
        """Save the prompt evolution report to a file"""
        report = self.generate_evolution_report(prompt_type, run_id)

        report_file = (
            self.path_manager.paths.optimized_prompts_dir
            / f"{self.path_manager.domain}_{prompt_type}_{run_id}_evolution_report.md"
        )

        with open(report_file, "w") as f:
            f.write(report)

        return report_file
