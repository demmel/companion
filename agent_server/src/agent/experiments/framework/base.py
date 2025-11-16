"""
Base abstractions for the experiment framework.

Key insight: Experiments vary across the VARIANT INTERFACE being tested,
not the data types. Test cases execute themselves using variants that
implement a specific interface.
"""

from typing import TypeVar, Generic, Optional, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel

# Type variable for variant interface (e.g., StructuredOutputFormat, LLMModel, etc.)
TVariant = TypeVar('TVariant')


class TestCase(ABC, Generic[TVariant]):
    """
    A test case that can execute itself using any variant implementing TVariant interface.

    Type parameters:
        TVariant: The variant interface type this test case expects
                 (e.g., StructuredOutputFormat, LLMProvider, etc.)

    Key design:
        - Test case knows its own data types (input, output, expected)
        - Test case knows how to use a variant to execute itself
        - Each test case can have completely different data types
        - All test cases in an experiment share the same variant interface

    Example:
        class FactExtractionTest(TestCase[StructuredOutputFormat]):
            def execute(self, variant: StructuredOutputFormat) -> ExtractionResponse:
                schema = variant.generate_schema(ExtractionResponse)
                prompt = variant.build_prompt(self.system_prompt, self.input_text, schema)
                response = call_llm(prompt)
                parsed = variant.parse_response(response)
                return ExtractionResponse.model_validate(parsed)

            def expected_output(self) -> ExtractionResponse:
                return ExtractionResponse(facts=[...])
    """

    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this test case.

        Used in directory structure and result aggregation.
        """
        ...

    @abstractmethod
    def execute(self, variant: TVariant) -> BaseModel:
        """
        Execute this test case using the given variant.

        The test case uses the variant's interface to produce output.
        Return type is BaseModel but actual type varies by test case.

        Args:
            variant: The variant to use for execution

        Returns:
            Output as a Pydantic BaseModel (actual type varies)

        Raises:
            Any exception if execution fails (will be caught by framework)
        """
        ...

    def expected_output(self) -> Optional[BaseModel]:
        """
        Optional ground truth for evaluation.

        Return type is BaseModel but actual type varies by test case.
        Should match the type returned by execute().

        Returns:
            Expected output for comparison, or None if no ground truth
        """
        return None


class MetricsCalculator(ABC):
    """
    Calculates metrics by comparing output to expected output.

    Works polymorphically with BaseModel - implementations can inspect
    actual types if needed.

    Example:
        class SemanticSimilarityCalculator(MetricsCalculator):
            def calculate(self, output: BaseModel, expected: Optional[BaseModel]) -> Dict[str, float]:
                if expected is None:
                    return {}

                # Both are BaseModel - serialize to JSON for comparison
                output_json = output.model_dump_json(sort_keys=True)
                expected_json = expected.model_dump_json(sort_keys=True)

                similarity = self._compute_embedding_similarity(output_json, expected_json)
                return {
                    "precision": similarity,
                    "recall": similarity,
                    "f1": similarity
                }
    """

    @abstractmethod
    def calculate(
        self,
        output: BaseModel,
        expected: Optional[BaseModel]
    ) -> Dict[str, float]:
        """
        Calculate metrics from output and expected.

        Args:
            output: The output from test case execution
            expected: The expected output from test case (may be None)

        Returns:
            Dictionary mapping metric names to numeric values
            Examples: {"f1": 0.85, "precision": 0.90, "recall": 0.80}
        """
        ...
