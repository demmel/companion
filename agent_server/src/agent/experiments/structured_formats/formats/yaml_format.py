"""
YAML format implementation.

Uses indentation-based YAML, potentially more natural for LLMs.
"""

import re
import yaml
from typing import Type, Dict, Any, Optional, get_origin, get_args
from pydantic import BaseModel, ValidationError

from ..base_format import StructuredOutputFormat


class YAMLFormat(StructuredOutputFormat):
    """YAML format for structured output."""

    def name(self) -> str:
        return "yaml"

    @property
    def max_nesting_depth(self) -> Optional[int]:
        return None  # Unlimited

    def generate_schema(self, model: Type[BaseModel]) -> str:
        """Generate YAML schema description from Pydantic model."""
        schema_lines = []
        schema_lines.extend(self._generate_field_schema(model, indent=0))
        return "\n".join(schema_lines)

    def _generate_field_schema(
        self, model: Type[BaseModel], indent: int = 0
    ) -> list[str]:
        """Recursively generate field schema."""
        lines = []
        prefix = " " * indent

        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            description = field_info.description or ""
            comment = f"  # {description}" if description else ""

            # Handle lists
            origin = get_origin(annotation)
            if origin is list:
                args = get_args(annotation)
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    # List of nested objects
                    lines.append(f"{prefix}{field_name}:{comment}")
                    lines.append(f"{prefix}  - # First item")
                    lines.extend(self._generate_field_schema(args[0], indent + 4))
                else:
                    # List of primitives
                    type_name = (
                        getattr(args[0], "__name__", str(args[0])) if args else "value"
                    )
                    lines.append(f"{prefix}{field_name}:{comment}")
                    lines.append(f"{prefix}  - value  # {type_name}")
            # Handle nested objects
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                lines.append(f"{prefix}{field_name}:{comment}")
                lines.extend(self._generate_field_schema(annotation, indent + 2))
            # Handle primitives
            else:
                type_name = getattr(annotation, "__name__", str(annotation))
                lines.append(f"{prefix}{field_name}: value{comment}")

        return lines

    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        """Build prompt with YAML schema instructions."""
        prompt_parts = [
            "You are a helpful AI assistant that provides structured responses.",
            "",
            "TASK:",
            system_prompt,
            "",
            "INPUT:",
            user_input,
            "",
            "RESPONSE FORMAT:",
            "You must respond with valid YAML data that follows this structure:",
            "",
            schema_str,
            "",
            "IMPORTANT:",
            "- Use the exact field names shown",
            "- Pay careful attention to indentation (use 2 spaces per level)",
            "- For lists, use '- ' prefix for each item",
            "- Include all required fields",
            "- Do not include any text before or after the YAML",
            "- Use proper YAML syntax (colons, dashes, indentation)",
        ]

        return "\n".join(prompt_parts)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse YAML from response."""
        response_text = response_text.strip()

        # Remove reasoning tags
        cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        cleaned_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", cleaned_text, flags=re.DOTALL
        )
        cleaned_text = cleaned_text.strip()

        # Remove markdown code blocks if present
        if cleaned_text.startswith("```yaml") or cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:yaml)?\s*\n", "", cleaned_text)
            cleaned_text = re.sub(r"\n```\s*$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        # Parse YAML
        try:
            parsed_data = yaml.safe_load(cleaned_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

        if not isinstance(parsed_data, dict):
            raise ValueError(
                f"YAML must parse to a dictionary, got {type(parsed_data)}"
            )

        return parsed_data

    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        """Convert Pydantic errors to clear, YAML-friendly messages."""
        error_messages = []

        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            input_value = err.get("input")

            # Format specific error types
            if error_type == "string_type" and isinstance(input_value, list):
                error_messages.append(
                    f"Field '{field_path}' must be a SINGLE STRING value, "
                    f"not a list. If you need multiple values, create separate list items with '- '."
                )
            elif error_type == "missing":
                error_messages.append(
                    f"Field '{field_path}' is REQUIRED and missing. "
                    f"Add this field with proper indentation."
                )
            elif input_value is None:
                error_messages.append(
                    f"Field '{field_path}' cannot be null/empty. "
                    f"Provide an actual value after the colon."
                )
            else:
                msg = err["msg"]
                error_messages.append(f"Field '{field_path}': {msg}")

        return "Validation errors:\n" + "\n".join(
            f"  - {msg}" for msg in error_messages
        )
