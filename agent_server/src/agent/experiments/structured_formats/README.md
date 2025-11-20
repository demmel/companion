# Structured Output Format Experiment

Experiment to compare different structured output formats (JSON, XML, YAML, S-Expressions, etc.) to determine which works best with local LLMs for structured data extraction.

## Overview

This experiment tests the hypothesis that different output formats may have varying success rates, error patterns, and performance characteristics when used with local quantized models.

### Key Features

- **Pluggable format system**: Easy to add new formats
- **Real use cases**: Tests based on actual codebase models (fact extraction, memory queries, action planning, etc.)
- **Statistical rigor**: Multiple runs per test case with significance testing
- **Comprehensive metrics**: Tracks success rate, F1 score, token usage, time, errors, and richness
- **Better error messages**: Each format provides LLM-friendly error formatting (not raw Pydantic errors)

## Formats Tested

1. **JSON** (baseline) - Standard JSON with Pydantic schema
2. **XML** - Tagged XML format
3. **YAML** - Indentation-based YAML
4. **S-Expressions** - Lisp-style s-expressions

Additional formats can be easily added by implementing the `StructuredOutputFormat` interface.

## Usage

### Quick Start

```bash
# Run experiment with default settings (15 runs per test case)
cd /home/demmel/projects/agent/agent_server
python -m agent.experiments.structured_formats.run_experiment

# Run with custom number of runs
python -m agent.experiments.structured_formats.run_experiment --num-runs 20

# Specify output directory
python -m agent.experiments.structured_formats.run_experiment --output-dir my_results
```

### Example Output

```
==================================================================================
STRUCTURED OUTPUT FORMAT EXPERIMENT RESULTS
==================================================================================

Formats tested: 4
Test cases: 7
Runs per test case: 15
Total LLM calls: 420

----------------------------------------------------------------------------------
OVERALL RANKINGS
----------------------------------------------------------------------------------

By Success Rate:
  1. yaml: 92.3% (F1=0.89, tokens=1150)
  2. sexp: 88.7% (F1=0.87, tokens=1200)
  3. xml: 85.2% (F1=0.91, tokens=1300)
  4. json: 82.1% (F1=0.83, tokens=1180)

By F1 Score:
  1. xml: 0.91 (success=85.2%, tokens=1300)
  2. yaml: 0.89 (success=92.3%, tokens=1150)
  3. sexp: 0.87 (success=88.7%, tokens=1200)
  4. json: 0.83 (success=82.1%, tokens=1180)

...
```

## Architecture

### Components

```
structured_formats/
├── base_format.py           # Abstract format interface
├── formats/                 # Format implementations
│   ├── json_format.py       # JSON (baseline)
│   ├── xml_format.py        # XML
│   ├── yaml_format.py       # YAML
│   └── sexp_format.py       # S-Expressions
├── test_cases.py            # Test cases with ground truth
├── metrics.py               # Metrics tracking with statistics
├── evaluation.py            # Correctness evaluation
├── experiment_runner.py     # Experiment orchestration
├── analysis.py              # Statistical analysis and reporting
└── run_experiment.py        # Main entry point
```

### Adding a New Format

1. Create a new file in `formats/` (e.g., `toml_format.py`)
2. Implement the `StructuredOutputFormat` interface:
   - `name`: Unique format name
   - `max_nesting_depth`: Maximum nesting depth (or None)
   - `generate_schema()`: Generate schema from Pydantic model
   - `build_prompt()`: Build LLM prompt with schema
   - `parse_response()`: Parse LLM output to dict
   - `format_error()`: Convert Pydantic errors to LLM-friendly messages
3. Add to `formats/__init__.py` and `run_experiment.py`

Example:

```python
from ..base_format import StructuredOutputFormat

class TOMLFormat(StructuredOutputFormat):
    @property
    def name(self) -> str:
        return "toml"

    @property
    def max_nesting_depth(self) -> Optional[int]:
        return None  # Unlimited

    def generate_schema(self, model: Type[BaseModel]) -> str:
        # Generate TOML schema description
        ...

    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        # Build prompt with TOML examples
        ...

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        # Parse TOML to dict
        import tomli
        return tomli.loads(response_text)

    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        # Format errors for TOML
        ...
```

### Adding Test Cases

Add to `test_cases.py`:

```python
test_cases.append(
    TestCase(
        name="my_test_case",
        model=MyPydanticModel,
        category="my_category",
        system_prompt="Extract data...",
        user_input="Complex realistic input text...",
        expected=MyPydanticModel(
            field1="expected_value",
            field2=123,
        ),
    )
)
```

## Metrics

### Tracked Metrics

- **Success Rate**: % of calls that produce valid output (with CI)
- **F1 Score**: Harmonic mean of precision/recall vs ground truth
- **Token Usage**: Input, output, and total tokens
- **Time**: Wall clock time (note: can be misleading due to caching)
- **Retries**: Average number of retries needed
- **Error Types**: Parse errors, validation errors, type errors, null errors
- **Richness**: % of optional fields populated

### Statistical Analysis

- Multiple runs per test case (default: 15)
- 95% confidence intervals
- Two-proportion z-test for success rates
- T-test for F1 scores and time
- Comparisons against JSON baseline

## Results

Results are saved to:
- `experiment_results/experiment_report.txt` - Human-readable report
- `experiment_results/experiment_results.json` - Machine-readable data

## Design Decisions

### Why Pluggable Formats?

The original `structured_llm.py` had formats hardcoded with if/else logic. This made it hard to:
- Experiment with new formats
- Provide format-specific error messages
- Test formats independently

The pluggable system makes each format self-contained and testable.

### Why Better Error Formatting?

Pydantic errors are notoriously unclear:
```
Input should be a valid string [type=string_type, input_value=['education', 'governance'], input_type=list]
```

Format-specific errors are clearer:
```
Field 'entities.aspects' must be a STRING like 'education', not an ARRAY ['education', 'governance'].
Choose one value or create multiple separate objects.
```

### Why Statistical Rigor?

Single runs can be misleading due to:
- Randomness in LLM sampling
- Caching effects
- GPU load variations

Multiple runs with statistical testing ensure we're measuring real differences, not noise.

## Future Work

- Implement additional formats (TOML, Markdown Tables, Python Dict Literals, etc.)
- Test with different models (Q8 quantization, different architectures)
- Test impact of few-shot examples
- Test impact of temperature settings
- Explore grammar-constrained generation (like Ollama's native schema support)
