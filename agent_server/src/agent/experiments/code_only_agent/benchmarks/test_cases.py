"""All benchmark test cases for code-only agent."""

from pathlib import Path
from typing import Optional

from agent.experiments.code_only_agent.benchmarks.base import (
    CodeAgentBenchmark,
    BenchmarkResult,
)
from agent.experiments.code_only_agent.state import AgentTurn, State


def count_function_calls(turn: AgentTurn) -> dict[str, int]:
    """Count how many times each function was called in the code."""
    counts: dict[str, int] = {}
    for iteration in turn.iterations:
        if iteration.code:
            for func in ["speak", "read_file", "list_files", "get_time"]:
                counts[func] = counts.get(func, 0) + iteration.code.count(f"{func}(")
    return counts


def get_errors(turn: AgentTurn) -> list[str]:
    """Extract errors from turn outputs."""
    from agent.experiments.code_only_agent.state import ErrorMessage

    errors = []
    for iteration in turn.iterations:
        for output in iteration.outputs:
            if isinstance(output, ErrorMessage):
                errors.append(output.error)
    return errors


# =============================================================================
# Category A: Basic Communication (6 tasks)
# =============================================================================


def validate_simple_greeting(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate simple_greeting benchmark."""
    speaks = turn.get_speaks()
    spoke = len(speaks) > 0
    has_greeting = any(
        word in " ".join(speaks).lower() for word in ["hello", "hi", "hey", "greetings"]
    )
    passed = spoke and has_greeting
    score = 1.0 if passed else (0.5 if spoke else 0.0)

    return BenchmarkResult(
        task_name="simple_greeting",
        category="communication",
        difficulty="easy",
        passed=passed,
        score=score,
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Agent spoke: {spoke}, Has greeting: {has_greeting}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_multi_part_message(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate multi_part_message benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Check for at least 2 facts (simplified - just check for multiple sentences/points)
    has_multiple_points = message.count(".") >= 2 or message.count("\n") >= 2
    passed = spoke and has_multiple_points

    return BenchmarkResult(
        task_name="multi_part_message",
        category="communication",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Multiple points: {has_multiple_points}",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_error_communication(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate error_communication benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    mentions_error = any(
        word in message
        for word in ["error", "not found", "doesn't exist", "cannot", "failed"]
    )
    passed = spoke and mentions_error

    return BenchmarkResult(
        task_name="error_communication",
        category="communication",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Mentions error: {mentions_error}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_clarification_request(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate clarification_request benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    asks_for_clarification = any(
        phrase in message
        for phrase in [
            "what",
            "which",
            "clarify",
            "specify",
            "unclear",
            "more information",
            "?",
        ]
    )
    passed = spoke and asks_for_clarification

    return BenchmarkResult(
        task_name="clarification_request",
        category="communication",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Asks for clarification: {asks_for_clarification}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_structured_response(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate structured_response benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    functions = ["speak", "read_file", "list_files", "get_time"]
    mentioned_count = sum(1 for func in functions if func in message)
    passed = spoke and mentioned_count >= 3

    return BenchmarkResult(
        task_name="structured_response",
        category="communication",
        difficulty="medium",
        passed=passed,
        score=mentioned_count / 4.0,
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Mentioned {mentioned_count}/4 functions",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_conversation_memory(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate conversation_memory benchmark."""
    # This test requires multi-turn - skipping for now
    passed = False
    return BenchmarkResult(
        task_name="conversation_memory",
        category="communication",
        difficulty="hard",
        passed=passed,
        score=0.0,
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details="Multi-turn test - not yet implemented",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


# =============================================================================
# Category B: Filesystem Operations (8 tasks)
# =============================================================================


def validate_list_current_directory(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate list_current_directory benchmark."""
    funcs = count_function_calls(turn)
    called_list_files = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    passed = called_list_files and spoke

    return BenchmarkResult(
        task_name="list_current_directory",
        category="filesystem",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if called_list_files else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Called list_files: {called_list_files}, Spoke: {spoke}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def setup_test_file(test_path: Path) -> None:
    """Create a test file."""
    test_file = test_path / "test_file.txt"
    test_file.write_text("This is test content.\nLine 2.\nLine 3.")


def validate_read_single_file(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate read_single_file benchmark."""
    funcs = count_function_calls(turn)
    called_read_file = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    contains_content = "test content" in message.lower()
    passed = called_read_file and spoke and contains_content

    return BenchmarkResult(
        task_name="read_single_file",
        category="filesystem",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if called_read_file else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read file: {called_read_file}, Spoke content: {contains_content}",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_count_files(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate count_files benchmark."""
    funcs = count_function_calls(turn)
    called_list_files = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Check if message contains a number
    has_number = any(char.isdigit() for char in message)
    passed = called_list_files and spoke and has_number

    return BenchmarkResult(
        task_name="count_files",
        category="filesystem",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if called_list_files else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {called_list_files}, Spoke number: {has_number}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_file_search(test_path: Path) -> None:
    """Create multiple files including target."""
    (test_path / "file1.txt").write_text("content 1")
    (test_path / "target.txt").write_text("target content")
    (test_path / "file2.txt").write_text("content 2")


def validate_file_search(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate file_search benchmark."""
    funcs = count_function_calls(turn)
    called_list_files = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    found_target = "target" in message and ("exist" in message or "found" in message)
    passed = called_list_files and spoke and found_target

    return BenchmarkResult(
        task_name="file_search",
        category="filesystem",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if called_list_files else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {called_list_files}, Found target: {found_target}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def setup_content_search(test_path: Path) -> None:
    """Create files with one containing keyword."""
    (test_path / "file1.txt").write_text("normal content")
    (test_path / "file2.txt").write_text("This has the secret_keyword inside")
    (test_path / "file3.txt").write_text("other content")


def validate_file_content_search(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate file_content_search benchmark."""
    funcs = count_function_calls(turn)
    read_count = funcs.get("read_file", 0)
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    identified_file2 = "file2" in message
    passed = read_count > 0 and spoke and identified_file2

    return BenchmarkResult(
        task_name="file_content_search",
        category="filesystem",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.3 if read_count > 0 else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read {read_count} files, Identified file2: {identified_file2}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_multiple_files(test_path: Path) -> None:
    """Create multiple txt files."""
    (test_path / "doc1.txt").write_text("Document one content")
    (test_path / "doc2.txt").write_text("Document two content")
    (test_path / "doc3.txt").write_text("Document three content")


def validate_read_multiple_files(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate read_multiple_files benchmark."""
    funcs = count_function_calls(turn)
    read_count = funcs.get("read_file", 0)
    spoke = len(turn.get_speaks()) > 0
    # Ideally reads all 3
    passed = read_count >= 3 and spoke

    return BenchmarkResult(
        task_name="read_multiple_files",
        category="filesystem",
        difficulty="hard",
        passed=passed,
        score=min(read_count / 3.0, 1.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read {read_count}/3 files, Spoke: {spoke}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_handle_read_error(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate handle_read_error benchmark."""
    funcs = count_function_calls(turn)
    tried_read = funcs.get("read_file", 0) > 0
    errors = get_errors(turn)
    got_error = len(errors) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    communicated_issue = any(
        word in message for word in ["error", "not found", "doesn't exist", "cannot"]
    )
    passed = tried_read and spoke and communicated_issue

    return BenchmarkResult(
        task_name="handle_read_error",
        category="filesystem",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if tried_read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=errors,
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Tried: {tried_read}, Communicated error: {communicated_issue}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_nested_directory(test_path: Path) -> None:
    """Create nested directory structure."""
    nested = test_path / "nested"
    nested.mkdir()
    (nested / "nested_file.txt").write_text("nested content")


def validate_nested_directory_exploration(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate nested_directory_exploration benchmark."""
    funcs = count_function_calls(turn)
    called_list = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    found_nested_file = "nested_file" in message
    passed = called_list and spoke and found_nested_file

    return BenchmarkResult(
        task_name="nested_directory_exploration",
        category="filesystem",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if called_list else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Called list: {called_list}, Found nested file: {found_nested_file}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# Category C: Time/System Operations (3 tasks)
# =============================================================================


def validate_get_current_time(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate get_current_time benchmark."""
    funcs = count_function_calls(turn)
    called_get_time = funcs.get("get_time", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Check for time-like pattern (contains numbers and colons or time words)
    has_time_info = ":" in message or any(
        word in message.lower() for word in ["time", "hour", "minute", "am", "pm"]
    )
    passed = called_get_time and spoke and has_time_info

    return BenchmarkResult(
        task_name="get_current_time",
        category="time_system",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if called_get_time else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Called get_time: {called_get_time}, Has time info: {has_time_info}",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_time_and_action(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate time_and_action benchmark."""
    funcs = count_function_calls(turn)
    called_time = funcs.get("get_time", 0) > 0
    called_list = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    passed = called_time and called_list and spoke

    return BenchmarkResult(
        task_name="time_and_action",
        category="time_system",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if (called_time or called_list) else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Time: {called_time}, List: {called_list}, Spoke: {spoke}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_time_formatting(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate time_formatting benchmark."""
    funcs = count_function_calls(turn)
    called_time = funcs.get("get_time", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    has_hour = any(char.isdigit() for char in message)
    passed = called_time and spoke and has_hour

    return BenchmarkResult(
        task_name="time_formatting",
        category="time_system",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if called_time else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Called time: {called_time}, Has hour: {has_hour}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# Category D: Multi-Step Reasoning (5 tasks - simplified set)
# =============================================================================


def validate_sequential_operations(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate sequential_operations benchmark."""
    funcs = count_function_calls(turn)
    listed = funcs.get("list_files", 0) > 0
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    # Check order by looking at iterations
    correct_order = False
    for i, iteration in enumerate(turn.iterations):
        if iteration.code is not None and "list_files" in iteration.code:
            # Check if later iteration has read_file
            for j in range(i + 1, len(turn.iterations)):
                later_code = turn.iterations[j].code
                if later_code is not None and "read_file" in later_code:
                    correct_order = True
                    break
    passed = listed and read and spoke and correct_order

    return BenchmarkResult(
        task_name="sequential_operations",
        category="multi_step",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if (listed and read) else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {listed}, Read: {read}, Correct order: {correct_order}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def setup_check_file(test_path: Path) -> None:
    """Create check.txt file."""
    (test_path / "check.txt").write_text("Check file content here")


def validate_conditional_logic(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate conditional_logic benchmark."""
    funcs = count_function_calls(turn)
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    has_content = "content" in message.lower() or "check" in message.lower()
    passed = read and spoke and has_content

    return BenchmarkResult(
        task_name="conditional_logic",
        category="multi_step",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read: {read}, Has content: {has_content}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_mixed_files(test_path: Path) -> None:
    """Create mix of txt and py files."""
    (test_path / "file1.txt").write_text("text 1")
    (test_path / "script.py").write_text("print('hello')")
    (test_path / "file2.txt").write_text("text 2")
    (test_path / "another.py").write_text("x = 1")


def validate_filtering_and_counting(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate filtering_and_counting benchmark."""
    funcs = count_function_calls(turn)
    listed = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Should say 2 txt files
    has_two = "2" in message or "two" in message.lower()
    passed = listed and spoke and has_two

    return BenchmarkResult(
        task_name="filtering_and_counting",
        category="multi_step",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if listed else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {listed}, Found 2: {has_two}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_numbered_files(test_path: Path) -> None:
    """Create files with line counts."""
    (test_path / "file1.txt").write_text("line1\nline2\nline3")
    (test_path / "file2.txt").write_text("line1\nline2")


def validate_aggregation(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate aggregation benchmark."""
    funcs = count_function_calls(turn)
    read_count = funcs.get("read_file", 0)
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Total should be 5 lines
    has_five = "5" in message or "five" in message.lower()
    passed = read_count >= 2 and spoke and has_five

    return BenchmarkResult(
        task_name="aggregation",
        category="multi_step",
        difficulty="hard",
        passed=passed,
        score=min(read_count / 2.0, 1.0) if spoke else 0.0,
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read {read_count}/2 files, Found total: {has_five}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_length_files(test_path: Path) -> None:
    """Create files of different lengths."""
    (test_path / "file1.txt").write_text("short")
    (test_path / "file2.txt").write_text("This is a much longer file with more content")


def validate_comparison(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate comparison benchmark."""
    funcs = count_function_calls(turn)
    read_count = funcs.get("read_file", 0)
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    # file2 is longer
    identified_longer = "file2" in message and "longer" in message
    passed = read_count >= 2 and spoke and identified_longer

    return BenchmarkResult(
        task_name="comparison",
        category="multi_step",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if read_count >= 2 else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read {read_count} files, Identified file2 longer: {identified_longer}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# Category E: Edge Cases (4 tasks - simplified)
# =============================================================================


def setup_empty_directory(test_path: Path) -> None:
    """Create empty directory."""
    (test_path / "empty_dir").mkdir()


def validate_empty_directory(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate empty_directory benchmark."""
    funcs = count_function_calls(turn)
    listed = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    mentions_empty = "empty" in message or "no files" in message or "0" in message
    passed = listed and spoke and mentions_empty

    return BenchmarkResult(
        task_name="empty_directory",
        category="edge_cases",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if listed else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {listed}, Mentions empty: {mentions_empty}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_empty_file(test_path: Path) -> None:
    """Create empty file."""
    (test_path / "empty.txt").write_text("")


def validate_empty_file(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate empty_file benchmark."""
    funcs = count_function_calls(turn)
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    passed = read and spoke

    return BenchmarkResult(
        task_name="empty_file",
        category="edge_cases",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read: {read}, Spoke: {spoke}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_ambiguous_instruction(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate ambiguous_instruction benchmark."""
    spoke = len(turn.get_speaks()) > 0
    no_errors = len(get_errors(turn)) == 0
    passed = spoke and no_errors

    return BenchmarkResult(
        task_name="ambiguous_instruction",
        category="edge_cases",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, No errors: {no_errors}",
        user_input=turn.user_input,
        agent_response=" ".join(turn.get_speaks()),
    )


def validate_impossible_task(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate impossible_task benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).lower()
    recognizes_limitation = any(
        phrase in message
        for phrase in ["cannot", "can't", "unable", "don't have", "not available"]
    )
    passed = spoke and recognizes_limitation

    return BenchmarkResult(
        task_name="impossible_task",
        category="edge_cases",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Recognizes limitation: {recognizes_limitation}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# Category F: Code Execution & Logic (5 tasks - simplified)
# =============================================================================


def validate_simple_arithmetic(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate simple_arithmetic benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    has_42 = "42" in message
    passed = spoke and has_42

    return BenchmarkResult(
        task_name="simple_arithmetic",
        category="code_logic",
        difficulty="easy",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Has 42: {has_42}",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_string_manipulation(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate string_manipulation benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    has_caps = "HELLO WORLD" in message
    passed = spoke and has_caps

    return BenchmarkResult(
        task_name="string_manipulation",
        category="code_logic",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Has uppercase: {has_caps}",
        user_input=turn.user_input,
        agent_response=message,
    )


def validate_list_processing(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate list_processing benchmark."""
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Check for numbers 1-10
    has_numbers = all(str(i) in message for i in range(1, 11))
    passed = spoke and has_numbers

    return BenchmarkResult(
        task_name="list_processing",
        category="code_logic",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if spoke else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=count_function_calls(turn),
        execution_time_seconds=0.0,
        details=f"Spoke: {spoke}, Has all numbers: {has_numbers}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_csv_file(test_path: Path) -> None:
    """Create CSV-like file."""
    (test_path / "data.csv").write_text("name,age,city\nAlice,30,NYC\nBob,25,LA")


def validate_data_structure_usage(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate data_structure_usage benchmark."""
    funcs = count_function_calls(turn)
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Second column is ages: 30, 25
    has_ages = "30" in message and "25" in message
    passed = read and spoke and has_ages

    return BenchmarkResult(
        task_name="data_structure_usage",
        category="code_logic",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read: {read}, Has ages: {has_ages}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_log_file(test_path: Path) -> None:
    """Create log file with ERROR lines."""
    (test_path / "log.txt").write_text(
        "INFO: Starting\nERROR: Connection failed\nINFO: Retrying\nERROR: Timeout\nINFO: Done"
    )


def validate_pattern_matching(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate pattern_matching benchmark."""
    funcs = count_function_calls(turn)
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks()).upper()
    # Should mention both ERROR lines
    has_connection = "CONNECTION" in message or "FAILED" in message
    has_timeout = "TIMEOUT" in message
    passed = read and spoke and has_connection and has_timeout

    return BenchmarkResult(
        task_name="pattern_matching",
        category="code_logic",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read: {read}, Found both errors: {has_connection and has_timeout}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# Category G: Integration (2 tasks - simplified)
# =============================================================================


def setup_inventory_files(test_path: Path) -> None:
    """Create various files for inventory."""
    (test_path / "doc.txt").write_text("text document")
    (test_path / "script.py").write_text("print('hello')")
    (test_path / "data.json").write_text('{"key": "value"}')


def validate_file_inventory(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate file_inventory benchmark."""
    funcs = count_function_calls(turn)
    listed = funcs.get("list_files", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Should mention all 3 files
    mentions_all = all(f in message for f in ["doc.txt", "script.py", "data.json"])
    passed = listed and spoke and mentions_all

    return BenchmarkResult(
        task_name="file_inventory",
        category="integration",
        difficulty="medium",
        passed=passed,
        score=1.0 if passed else (0.5 if listed else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Listed: {listed}, Mentions all: {mentions_all}",
        user_input=turn.user_input,
        agent_response=message,
    )


def setup_log_analysis(test_path: Path) -> None:
    """Create server log."""
    (test_path / "server.log").write_text(
        "ERROR: Failed\nINFO: Started\nERROR: Crash\nINFO: OK\nINFO: Done"
    )


def validate_log_analysis(
    turn: AgentTurn, state: State, test_path: Path
) -> BenchmarkResult:
    """Validate log_analysis benchmark."""
    funcs = count_function_calls(turn)
    read = funcs.get("read_file", 0) > 0
    spoke = len(turn.get_speaks()) > 0
    message = " ".join(turn.get_speaks())
    # Should identify 2 errors and 3 info
    has_2_errors = "2" in message or "two" in message.lower()
    has_3_info = "3" in message or "three" in message.lower()
    passed = read and spoke and has_2_errors and has_3_info

    return BenchmarkResult(
        task_name="log_analysis",
        category="integration",
        difficulty="hard",
        passed=passed,
        score=1.0 if passed else (0.5 if read else 0.0),
        iterations_used=len(turn.iterations),
        max_iterations=10,
        errors_encountered=get_errors(turn),
        functions_called=funcs,
        execution_time_seconds=0.0,
        details=f"Read: {read}, Found 2 errors: {has_2_errors}, Found 3 info: {has_3_info}",
        user_input=turn.user_input,
        agent_response=message,
    )


# =============================================================================
# All Benchmarks Registry
# =============================================================================


def get_all_benchmarks() -> list[CodeAgentBenchmark]:
    """Return all benchmark test cases."""
    return [
        # Category A: Basic Communication
        CodeAgentBenchmark(
            name="simple_greeting",
            category="communication",
            difficulty="easy",
            description="Agent should respond with a greeting",
            user_input="Say hello",
            setup_fn=None,
            validation_fn=validate_simple_greeting,
        ),
        CodeAgentBenchmark(
            name="multi_part_message",
            category="communication",
            difficulty="easy",
            description="Agent should provide multiple facts",
            user_input="Tell me three facts about Python",
            setup_fn=None,
            validation_fn=validate_multi_part_message,
        ),
        CodeAgentBenchmark(
            name="error_communication",
            category="communication",
            difficulty="medium",
            description="Agent should handle errors gracefully",
            user_input="Read the file /nonexistent/file.txt and tell me what it says",
            setup_fn=None,
            validation_fn=validate_error_communication,
        ),
        CodeAgentBenchmark(
            name="clarification_request",
            category="communication",
            difficulty="medium",
            description="Agent should ask for clarification",
            user_input="Do the thing",
            setup_fn=None,
            validation_fn=validate_clarification_request,
        ),
        CodeAgentBenchmark(
            name="structured_response",
            category="communication",
            difficulty="medium",
            description="Agent should list available functions",
            user_input="List the available functions and explain each one",
            setup_fn=None,
            validation_fn=validate_structured_response,
        ),
        CodeAgentBenchmark(
            name="conversation_memory",
            category="communication",
            difficulty="hard",
            description="Agent should remember previous conversation",
            user_input="What's my favorite color?",
            setup_fn=None,
            validation_fn=validate_conversation_memory,
        ),
        # Category B: Filesystem Operations
        CodeAgentBenchmark(
            name="list_current_directory",
            category="filesystem",
            difficulty="easy",
            description="Agent should list files",
            user_input="List all files in the current directory",
            setup_fn=None,
            validation_fn=validate_list_current_directory,
        ),
        CodeAgentBenchmark(
            name="count_files",
            category="filesystem",
            difficulty="easy",
            description="Agent should count files",
            user_input="How many files are in the current directory?",
            setup_fn=None,
            validation_fn=validate_count_files,
        ),
        CodeAgentBenchmark(
            name="read_single_file",
            category="filesystem",
            difficulty="medium",
            description="Agent should read a file",
            user_input="Read test_file.txt and tell me what it says",
            setup_fn=setup_test_file,
            validation_fn=validate_read_single_file,
        ),
        CodeAgentBenchmark(
            name="file_search",
            category="filesystem",
            difficulty="medium",
            description="Agent should find a specific file",
            user_input="Find a file named 'target.txt' and tell me if it exists",
            setup_fn=setup_file_search,
            validation_fn=validate_file_search,
        ),
        CodeAgentBenchmark(
            name="file_content_search",
            category="filesystem",
            difficulty="hard",
            description="Agent should search file contents",
            user_input="Find which file contains the word 'secret_keyword'",
            setup_fn=setup_content_search,
            validation_fn=validate_file_content_search,
        ),
        CodeAgentBenchmark(
            name="read_multiple_files",
            category="filesystem",
            difficulty="hard",
            description="Agent should read multiple files",
            user_input="Read all .txt files in the current directory and summarize each",
            setup_fn=setup_multiple_files,
            validation_fn=validate_read_multiple_files,
        ),
        CodeAgentBenchmark(
            name="handle_read_error",
            category="filesystem",
            difficulty="medium",
            description="Agent should handle read errors",
            user_input="Read the file that doesn't exist at /fake/path.txt",
            setup_fn=None,
            validation_fn=validate_handle_read_error,
        ),
        CodeAgentBenchmark(
            name="nested_directory_exploration",
            category="filesystem",
            difficulty="hard",
            description="Agent should explore nested directories",
            user_input="List files in the subdirectory called 'nested'",
            setup_fn=setup_nested_directory,
            validation_fn=validate_nested_directory_exploration,
        ),
        # Category C: Time/System
        CodeAgentBenchmark(
            name="get_current_time",
            category="time_system",
            difficulty="easy",
            description="Agent should get current time",
            user_input="What time is it?",
            setup_fn=None,
            validation_fn=validate_get_current_time,
        ),
        CodeAgentBenchmark(
            name="time_and_action",
            category="time_system",
            difficulty="medium",
            description="Agent should combine time with other operations",
            user_input="What time is it, and how many files are in the current directory?",
            setup_fn=None,
            validation_fn=validate_time_and_action,
        ),
        CodeAgentBenchmark(
            name="time_formatting",
            category="time_system",
            difficulty="medium",
            description="Agent should extract hour from time",
            user_input="Tell me the current hour of the day",
            setup_fn=None,
            validation_fn=validate_time_formatting,
        ),
        # Category D: Multi-Step Reasoning
        CodeAgentBenchmark(
            name="sequential_operations",
            category="multi_step",
            difficulty="medium",
            description="Agent should perform operations in sequence",
            user_input="First list files, then read the first one you find",
            setup_fn=setup_test_file,
            validation_fn=validate_sequential_operations,
        ),
        CodeAgentBenchmark(
            name="conditional_logic",
            category="multi_step",
            difficulty="hard",
            description="Agent should use conditional logic",
            user_input="If check.txt exists, read it. Otherwise, tell me it doesn't exist.",
            setup_fn=setup_check_file,
            validation_fn=validate_conditional_logic,
        ),
        CodeAgentBenchmark(
            name="filtering_and_counting",
            category="multi_step",
            difficulty="hard",
            description="Agent should filter and count",
            user_input="Count how many .txt files are in the current directory",
            setup_fn=setup_mixed_files,
            validation_fn=validate_filtering_and_counting,
        ),
        CodeAgentBenchmark(
            name="aggregation",
            category="multi_step",
            difficulty="hard",
            description="Agent should aggregate data from multiple files",
            user_input="Read all files and tell me the total number of lines across all files",
            setup_fn=setup_numbered_files,
            validation_fn=validate_aggregation,
        ),
        CodeAgentBenchmark(
            name="comparison",
            category="multi_step",
            difficulty="hard",
            description="Agent should compare files",
            user_input="Which file is longer: file1.txt or file2.txt?",
            setup_fn=setup_length_files,
            validation_fn=validate_comparison,
        ),
        # Category E: Edge Cases
        CodeAgentBenchmark(
            name="empty_directory",
            category="edge_cases",
            difficulty="easy",
            description="Agent should handle empty directories",
            user_input="List files in the empty_dir directory",
            setup_fn=setup_empty_directory,
            validation_fn=validate_empty_directory,
        ),
        CodeAgentBenchmark(
            name="empty_file",
            category="edge_cases",
            difficulty="easy",
            description="Agent should handle empty files",
            user_input="Read empty.txt",
            setup_fn=setup_empty_file,
            validation_fn=validate_empty_file,
        ),
        CodeAgentBenchmark(
            name="ambiguous_instruction",
            category="edge_cases",
            difficulty="hard",
            description="Agent should handle ambiguous instructions",
            user_input="Do something useful",
            setup_fn=None,
            validation_fn=validate_ambiguous_instruction,
        ),
        CodeAgentBenchmark(
            name="impossible_task",
            category="edge_cases",
            difficulty="hard",
            description="Agent should recognize impossible tasks",
            user_input="Delete all files",
            setup_fn=None,
            validation_fn=validate_impossible_task,
        ),
        # Category F: Code Execution & Logic
        CodeAgentBenchmark(
            name="simple_arithmetic",
            category="code_logic",
            difficulty="easy",
            description="Agent should perform arithmetic",
            user_input="What is 15 + 27?",
            setup_fn=None,
            validation_fn=validate_simple_arithmetic,
        ),
        CodeAgentBenchmark(
            name="string_manipulation",
            category="code_logic",
            difficulty="medium",
            description="Agent should manipulate strings",
            user_input="Tell me 'hello world' in all capital letters",
            setup_fn=None,
            validation_fn=validate_string_manipulation,
        ),
        CodeAgentBenchmark(
            name="list_processing",
            category="code_logic",
            difficulty="medium",
            description="Agent should process lists",
            user_input="Count from 1 to 10",
            setup_fn=None,
            validation_fn=validate_list_processing,
        ),
        CodeAgentBenchmark(
            name="data_structure_usage",
            category="code_logic",
            difficulty="medium",
            description="Agent should parse structured data",
            user_input="Read data.csv and tell me the values in the second column",
            setup_fn=setup_csv_file,
            validation_fn=validate_data_structure_usage,
        ),
        CodeAgentBenchmark(
            name="pattern_matching",
            category="code_logic",
            difficulty="hard",
            description="Agent should match patterns in text",
            user_input="Read log.txt and tell me all lines that contain 'ERROR'",
            setup_fn=setup_log_file,
            validation_fn=validate_pattern_matching,
        ),
        # Category G: Integration
        CodeAgentBenchmark(
            name="file_inventory",
            category="integration",
            difficulty="medium",
            description="Agent should provide comprehensive file inventory",
            user_input="Give me a complete inventory of this directory: files and their types",
            setup_fn=setup_inventory_files,
            validation_fn=validate_file_inventory,
        ),
        CodeAgentBenchmark(
            name="log_analysis",
            category="integration",
            difficulty="hard",
            description="Agent should analyze log files",
            user_input="Read server.log and tell me how many errors and how many info messages there are",
            setup_fn=setup_log_analysis,
            validation_fn=validate_log_analysis,
        ),
    ]
