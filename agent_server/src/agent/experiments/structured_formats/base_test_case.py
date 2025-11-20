"""
Base test case for structured output format experiments.

Provides common execution logic for testing different output formats.
"""

import time
import logging
from typing import Type, Optional
from pydantic import BaseModel, ValidationError

from agent.llm import LLM, SupportedModel
from agent.experiments.framework import TestCase

from .base_format import StructuredOutputFormat

logger = logging.getLogger(__name__)


class StructuredFormatTestCase(TestCase[StructuredOutputFormat]):
    """
    Base test case for structured format experiments.

    Concrete test cases inherit from this and provide:
    - model type
    - system prompt
    - user input
    - expected output

    This base class handles:
    - LLM execution
    - Schema generation
    - Response parsing
    - Validation with retries
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        max_retries: int = 3,
    ):
        """
        Initialize test case with LLM configuration.

        Args:
            llm: LLM instance for generation
            model: Model to use
            max_retries: Maximum retry attempts on validation errors
        """
        self.llm = llm
        self.model = model
        self.max_retries = max_retries

    def get_model_type(self) -> Type[BaseModel]:
        """Get the Pydantic model type for this test case."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Get the system prompt for this test case."""
        raise NotImplementedError

    def get_user_input(self) -> str:
        """Get the user input for this test case."""
        raise NotImplementedError

    def get_category(self) -> str:
        """Get the category for this test case (e.g., 'fact_extraction')."""
        raise NotImplementedError

    def execute(self, variant: StructuredOutputFormat) -> BaseModel:
        """
        Execute this test case using the given format variant.

        Args:
            variant: The structured output format to use

        Returns:
            Parsed and validated output as BaseModel

        Raises:
            Exception: If execution fails after all retries
        """
        model_type = self.get_model_type()
        system_prompt = self.get_system_prompt()
        user_input = self.get_user_input()

        # Validate compatibility
        variant.validate_model_compatibility(model_type)

        # Generate schema
        schema_str = variant.generate_schema(model_type)

        # Build initial prompt
        base_prompt = variant.build_prompt(system_prompt, user_input, schema_str)

        last_error_msg = None

        # Retry loop for validation errors
        for attempt in range(self.max_retries + 1):
            try:
                # Build prompt with error feedback if retrying
                if last_error_msg and attempt > 0:
                    prompt = f"{base_prompt}\n\nPrevious attempt had validation errors: {last_error_msg}\n\nPlease provide a corrected response."
                else:
                    prompt = base_prompt

                # Call LLM with direct generation
                response = self.llm.generate_complete(
                    model=self.model,
                    prompt=prompt,
                    num_predict=4096,
                    caller=f"format_experiment_{variant.name()}",
                    temperature=0.1,
                )

                # Parse response
                try:
                    parsed_data = variant.parse_response(response)
                except Exception as e:
                    logger.warning(f"Parse error on attempt {attempt + 1}: {e}")
                    raise ValueError(f"Parse error: {e}")

                # Validate with Pydantic
                try:
                    result = model_type.model_validate(parsed_data)
                    return result  # Success!

                except ValidationError as e:
                    # Format error for retry
                    last_error_msg = variant.format_error(e, model_type)

                    if attempt < self.max_retries:
                        logger.warning(
                            f"Validation error on attempt {attempt + 1}: {last_error_msg}"
                        )
                        continue
                    else:
                        raise

            except Exception as e:
                if attempt == self.max_retries:
                    raise
                logger.debug(f"Attempt {attempt + 1} failed: {str(e)}")
                continue

        raise RuntimeError("Unreachable - should have raised in loop")


class SimpleStructuredFormatTestCase(StructuredFormatTestCase):
    """
    Simple concrete test case that wraps static data.

    Useful for creating test cases from existing data without
    creating a new class for each one.
    """

    def __init__(
        self,
        name: str,
        model_type: Type[BaseModel],
        system_prompt: str,
        user_input: str,
        expected: BaseModel,
        category: str,
        llm: LLM,
        model: SupportedModel,
        max_retries: int = 3,
    ):
        """
        Initialize simple test case.

        Args:
            name: Test case name
            model_type: Pydantic model type
            system_prompt: System prompt for LLM
            user_input: User input text
            expected: Expected output
            category: Test category (e.g., 'fact_extraction')
            llm: LLM instance
            model: Model to use
            max_retries: Max retry attempts
        """
        super().__init__(llm, model, max_retries)
        self._name = name
        self._model_type = model_type
        self._system_prompt = system_prompt
        self._user_input = user_input
        self._expected = expected
        self._category = category

    def name(self) -> str:
        return self._name

    def get_model_type(self) -> Type[BaseModel]:
        return self._model_type

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_user_input(self) -> str:
        return self._user_input

    def expected_output(self) -> BaseModel:
        return self._expected

    def get_category(self) -> str:
        return self._category
