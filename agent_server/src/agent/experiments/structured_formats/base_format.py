"""
Base interface for structured output formats.

Defines the contract that all format implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, ValidationError


class StructuredOutputFormat(ABC):
    """
    Base class for structured output format implementations.

    Each format handles schema generation, prompt building, response parsing,
    and error formatting for LLM-friendly feedback.
    """

    @abstractmethod
    def name(self) -> str:
        """Unique name for this format (e.g., 'json', 'xml', 'yaml')"""
        pass

    @property
    @abstractmethod
    def max_nesting_depth(self) -> Optional[int]:
        """
        Maximum nesting depth supported by this format.

        Returns:
            None for unlimited depth, or integer for maximum depth.
            Used to validate model compatibility before running experiments.
        """
        pass

    @abstractmethod
    def generate_schema(self, model: Type[BaseModel]) -> str:
        """
        Generate schema description from Pydantic model.

        Args:
            model: Pydantic model class to generate schema for

        Returns:
            String representation of schema in this format
        """
        pass

    @abstractmethod
    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        """
        Build complete prompt with schema instructions.

        Args:
            system_prompt: System-level instructions
            user_input: User input/query
            schema_str: Schema string from generate_schema()

        Returns:
            Complete prompt ready for LLM
        """
        pass

    @abstractmethod
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into dictionary for Pydantic validation.

        Args:
            response_text: Raw LLM response text

        Returns:
            Dictionary that can be passed to Pydantic model_validate()

        Raises:
            ValueError: If response cannot be parsed
        """
        pass

    @abstractmethod
    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        """
        Convert Pydantic validation error to LLM-friendly message.

        Args:
            error: Pydantic ValidationError
            model: Pydantic model that failed validation

        Returns:
            Clear, actionable error message for LLM retry

        Example:
            Instead of: "Input should be a valid string [type=string_type, input_value=['x','y']]"
            Return: "Field 'entities.aspects' must be a STRING like 'education', not an ARRAY ['education', 'governance']"
        """
        pass

    def validate_model_compatibility(self, model: Type[BaseModel]) -> None:
        """
        Validate that model's nesting depth is compatible with this format.

        Args:
            model: Pydantic model to validate

        Raises:
            ValueError: If model nesting exceeds max_nesting_depth
        """
        if self.max_nesting_depth is None:
            return  # Unlimited depth

        actual_depth = self._calculate_model_depth(model)
        if actual_depth > self.max_nesting_depth:
            raise ValueError(
                f"Model has nesting depth {actual_depth} but {self.name} format "
                f"only supports depth {self.max_nesting_depth}"
            )

    def _calculate_model_depth(
        self, model: Type[BaseModel], current_depth: int = 1
    ) -> int:
        """
        Calculate maximum nesting depth of a Pydantic model.

        Args:
            model: Pydantic model class
            current_depth: Current recursion depth

        Returns:
            Maximum nesting depth
        """
        max_depth = current_depth

        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation

            # Handle Optional, List, etc.
            if hasattr(annotation, "__origin__"):
                args = getattr(annotation, "__args__", ())
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        depth = self._calculate_model_depth(arg, current_depth + 1)
                        max_depth = max(max_depth, depth)
            # Handle direct BaseModel fields
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                depth = self._calculate_model_depth(annotation, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    def __str__(self) -> str:
        return self.name()

    def __repr__(self) -> str:
        depth_str = (
            str(self.max_nesting_depth) if self.max_nesting_depth else "unlimited"
        )
        return f"{self.__class__.__name__}(name='{self.name()}', max_depth={depth_str})"
