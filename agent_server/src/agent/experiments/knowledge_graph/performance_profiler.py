#!/usr/bin/env python3
"""
Section-Level Performance Profiler for Knowledge Graph Operations

Provides detailed timing instrumentation to identify performance bottlenecks
across different sections of knowledge graph building.
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SectionTiming:
    """Timing data for a specific section"""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    def add_timing(self, duration: float) -> None:
        """Add a timing measurement"""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

    @property
    def avg_time(self) -> float:
        """Average time per call"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "total_time": self.total_time,
            "call_count": self.call_count,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "times": self.times[-10:],  # Last 10 measurements for analysis
        }


class PerformanceProfiler:
    """Comprehensive performance profiler for knowledge graph operations"""

    def __init__(self):
        self.sections: Dict[str, SectionTiming] = defaultdict(lambda: SectionTiming(""))
        self.session_start_time = time.time()
        self.active_sections: Dict[str, float] = {}  # For nested timing

    @contextmanager
    def section(self, section_name: str):
        """Context manager for timing a section of code"""
        start_time = time.time()

        # Handle nested sections
        if section_name in self.active_sections:
            section_name = f"{section_name}_nested_{len(self.active_sections)}"

        self.active_sections[section_name] = start_time

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Remove from active sections
            self.active_sections.pop(section_name, None)

            # Record timing
            if section_name not in self.sections:
                self.sections[section_name] = SectionTiming(section_name)

            self.sections[section_name].add_timing(duration)

            logger.debug(f"Section '{section_name}' completed in {duration:.3f}s")

    def get_breakdown_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance breakdown report"""

        total_session_time = time.time() - self.session_start_time
        total_measured_time = sum(
            section.total_time for section in self.sections.values()
        )

        # Sort sections by total time (highest first)
        sorted_sections = sorted(
            self.sections.values(), key=lambda s: s.total_time, reverse=True
        )

        breakdown = {
            "session_total_time": total_session_time,
            "measured_total_time": total_measured_time,
            "unmeasured_time": total_session_time - total_measured_time,
            "unmeasured_percentage": (
                ((total_session_time - total_measured_time) / total_session_time * 100)
                if total_session_time > 0
                else 0
            ),
            "sections": {},
        }

        for section in sorted_sections:
            section_stats = section.get_stats()
            section_stats["percentage_of_total"] = (
                (section.total_time / total_session_time * 100)
                if total_session_time > 0
                else 0
            )
            breakdown["sections"][section.name] = section_stats

        return breakdown

    def print_breakdown_report(self) -> None:
        """Print a formatted performance breakdown report"""

        report = self.get_breakdown_report()

        print("\n" + "=" * 80)
        print("📊 KNOWLEDGE GRAPH PERFORMANCE BREAKDOWN")
        print("=" * 80)

        print(f"Session Total Time: {report['session_total_time']:.2f}s")
        print(f"Measured Time: {report['measured_total_time']:.2f}s")
        print(
            f"Unmeasured Time: {report['unmeasured_time']:.2f}s ({report['unmeasured_percentage']:.1f}%)"
        )

        print(f"\n🔍 SECTION BREAKDOWN:")
        print("-" * 80)

        for section_name, stats in report["sections"].items():
            print(
                f"{section_name:.<40} {stats['total_time']:>8.2f}s ({stats['percentage_of_total']:>5.1f}%)"
            )
            print(
                f"{'':.<40} {stats['call_count']:>4} calls, avg: {stats['avg_time']:>6.3f}s"
            )

            if stats["call_count"] > 1:
                print(
                    f"{'':.<40} min: {stats['min_time']:>6.3f}s, max: {stats['max_time']:>6.3f}s"
                )
            print()

        # Identify bottlenecks
        print("🚨 PERFORMANCE BOTTLENECKS:")
        print("-" * 40)

        bottlenecks = [
            (name, stats)
            for name, stats in report["sections"].items()
            if stats["percentage_of_total"] > 10
        ]

        if bottlenecks:
            for name, stats in bottlenecks:
                print(f"• {name}: {stats['percentage_of_total']:.1f}% of total time")
        else:
            print("• No major bottlenecks (>10% of total time) identified")

        print("\n" + "=" * 80)

    def reset(self) -> None:
        """Reset all profiling data"""
        self.sections.clear()
        self.active_sections.clear()
        self.session_start_time = time.time()


# Global profiler instance
profiler = PerformanceProfiler()
