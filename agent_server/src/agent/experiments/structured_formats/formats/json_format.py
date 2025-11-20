"""
JSON format implementation (baseline).

Uses standard JSON with Pydantic JSON schema.
"""

import json
import re
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from ..base_format import StructuredOutputFormat


class JSONFormat(StructuredOutputFormat):
    """Standard JSON format with Pydantic schema."""

    def name(self) -> str:
        return "json"

    @property
    def max_nesting_depth(self) -> Optional[int]:
        return None  # Unlimited

    def generate_schema(self, model: Type[BaseModel]) -> str:
        """Generate JSON schema from Pydantic model."""
        schema = model.model_json_schema()
        return json.dumps(schema, indent=2)

    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        """Build prompt with JSON schema instructions."""
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
            "You must respond with valid JSON data that conforms to this schema:",
            "",
            schema_str,
            "",
            "IMPORTANT:",
            "- Create actual data, NOT the schema itself",
            "- Respond ONLY with a JSON object containing the actual values",
            "- Include all required fields with real content",
            "- Follow the field descriptions to generate appropriate values",
            "- Do not include any text before or after the JSON",
            "- Do not return the schema - return data that matches the schema",
        ]

        return "\n".join(prompt_parts)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        response_text = response_text.strip()

        # Remove reasoning tags that could contain misleading JSON-like content
        cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        cleaned_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", cleaned_text, flags=re.DOTALL
        )
        cleaned_text = cleaned_text.strip()

        # Extract JSON
        json_text = self._extract_json(cleaned_text)
        if not json_text:
            raise ValueError("No valid JSON found in response")

        # Fix escaping issues
        fixed_json = self._fix_json_escaping(json_text)

        # Parse JSON
        parsed_data = json.loads(fixed_json)

        # Handle common LLM response pattern where fields are wrapped in "properties"
        if isinstance(parsed_data, dict) and "properties" in parsed_data:
            parsed_data = parsed_data["properties"]

        return parsed_data

    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        """Convert Pydantic errors to clear, LLM-friendly messages."""
        error_messages = []

        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            input_value = err.get("input")

            # Format specific error types more clearly
            if error_type == "string_type" and isinstance(input_value, list):
                # Array instead of string
                value_repr = json.dumps(input_value)
                error_messages.append(
                    f"Field '{field_path}' must be a STRING (single value), "
                    f"not an ARRAY {value_repr}. "
                    f"Choose one value or create multiple separate objects."
                )
            elif error_type == "missing":
                # Missing required field
                error_messages.append(
                    f"Field '{field_path}' is REQUIRED and cannot be omitted. "
                    f"You must provide a value for this field."
                )
            elif "NoneType" in str(input_value) or input_value is None:
                # Null for required field
                error_messages.append(
                    f"Field '{field_path}' cannot be null. "
                    f"You must provide an actual value (the field is required)."
                )
            else:
                # Generic error with better formatting
                msg = err["msg"]
                error_messages.append(f"Field '{field_path}': {msg}")

        return "Validation errors:\n" + "\n".join(
            f"  - {msg}" for msg in error_messages
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling nested braces."""
        if text.startswith("{") and text.endswith("}"):
            return text

        # Look for JSON within the text
        if "{" in text and "}" in text:
            start_idx = text.find("{")

            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx : i + 1]

        return None

    def _fix_json_escaping(self, json_text: str) -> str:
        """Fix unescaped newlines and control characters in JSON string values."""

        def fix_string_content(match):
            key_part = match.group(1)  # The "key": part
            quote = match.group(2)  # Opening quote
            content = match.group(3)  # String content

            # Escape the content properly for JSON
            escaped_content = (
                content.replace("\\", "\\\\")  # Escape backslashes first
                .replace(quote, "\\" + quote)  # Escape quotes
                .replace("\n", "\\n")  # Escape newlines
                .replace("\r", "\\r")  # Escape carriage returns
                .replace("\t", "\\t")  # Escape tabs
                .replace("\b", "\\b")  # Escape backspace
                .replace("\f", "\\f")  # Escape form feed
            )

            # Handle other control characters
            escaped_content = re.sub(
                r"[\x00-\x1f\x7f]",
                lambda m: f"\\u{ord(m.group(0)):04x}",
                escaped_content,
            )

            return f"{key_part}{quote}{escaped_content}{quote}"

        # Pattern to match JSON key-value pairs with string values
        pattern = r'(\s*"[^"]*"\s*:\s*)(")([^"]*?)(?<!\\)"'

        # Apply the fix
        fixed = re.sub(pattern, fix_string_content, json_text, flags=re.DOTALL)

        return fixed
