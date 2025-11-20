"""
S-Expression format implementation.

Uses Lisp-style s-expressions with unambiguous nesting via parentheses.
"""

import re
from typing import Type, Dict, Any, Optional, List, Union, get_origin, get_args
from pydantic import BaseModel, ValidationError

from ..base_format import StructuredOutputFormat


class SExpFormat(StructuredOutputFormat):
    """S-Expression (Lisp-style) format for structured output."""

    def name(self) -> str:
        return "sexp"

    @property
    def max_nesting_depth(self) -> Optional[int]:
        return None  # Unlimited

    def generate_schema(self, model: Type[BaseModel]) -> str:
        """Generate S-Expression schema description."""
        schema_lines = [f"({model.__name__}"]
        schema_lines.extend(self._generate_field_schema(model, indent=2))
        schema_lines.append(")")
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
            comment = f"  ; {description}" if description else ""

            # Handle dicts (special case - show as nested structure)
            origin = get_origin(annotation)
            if origin is dict:
                lines.append(f"{prefix}({field_name}{comment}")
                lines.append(f'{prefix}  (key1 "value1")')
                lines.append(f'{prefix}  (key2 "value2")')
                lines.append(f"{prefix})")
            # Handle lists
            elif origin is list:
                args = get_args(annotation)
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    # List of nested objects
                    lines.append(f"{prefix}({field_name}{comment}")
                    lines.append(f"{prefix}  ({args[0].__name__}")
                    lines.extend(self._generate_field_schema(args[0], indent + 4))
                    lines.append(f"{prefix}  )")
                    lines.append(f"{prefix})")
                else:
                    # List of primitives
                    lines.append(
                        f'{prefix}({field_name} "value1" "value2" ...){comment}'
                    )
            # Handle nested objects
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                lines.append(f"{prefix}({field_name}{comment}")
                lines.extend(self._generate_field_schema(annotation, indent + 2))
                lines.append(f"{prefix})")
            # Handle primitives
            else:
                type_name = getattr(annotation, "__name__", str(annotation))
                lines.append(f'{prefix}({field_name} "value"){comment}')

        return lines

    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        """Build prompt with S-Expression schema instructions."""
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
            "You must respond with valid S-Expression (Lisp-style) data following this structure:",
            "",
            schema_str,
            "",
            "IMPORTANT:",
            "- Use exact field names as shown",
            '- String values must be in double quotes: "like this"',
            "- For dict fields, each pair must have EXACTLY 2 elements (key and single value):",
            '  (fieldname (key1 "val1") (key2 "val2"))',
            '  WRONG: (key "val1" "val2") - each key can only have ONE value',
            '- For lists, use multiple values: (fieldname "item1" "item2")',
            "- For nested objects, use: (fieldname (ObjectType ...))",
            "- Ensure all parentheses are balanced",
            "- Do not include any text before or after the S-Expression",
            "- Do NOT use Python dict syntax like {'key': 'value'}",
        ]

        return "\n".join(prompt_parts)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse S-Expression from response."""
        response_text = response_text.strip()

        # Remove reasoning tags
        cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        cleaned_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", cleaned_text, flags=re.DOTALL
        )
        cleaned_text = cleaned_text.strip()

        # Remove markdown code blocks if present
        if cleaned_text.startswith("```lisp") or cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:lisp|scheme)?\s*\n", "", cleaned_text)
            cleaned_text = re.sub(r"\n```\s*$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        # Remove comments (lines starting with ;)
        cleaned_text = re.sub(r";.*$", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = cleaned_text.strip()

        # Parse S-Expression
        try:
            tokens = self._tokenize(cleaned_text)
            parsed, _ = self._parse_tokens(tokens, 0)
        except Exception as e:
            raise ValueError(f"Invalid S-Expression: {e}")

        # Convert to dict
        if not isinstance(parsed, list) or len(parsed) < 1:
            raise ValueError("S-Expression must be a list with at least a name")

        return self._sexp_to_dict(parsed)

    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        """Convert Pydantic errors to clear, S-Expression-friendly messages."""
        error_messages = []

        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            input_value = err.get("input")

            # Format specific error types
            if error_type == "string_type" and isinstance(input_value, list):
                error_messages.append(
                    f"Field '{field_path}' must be a SINGLE quoted string: (field \"value\"), "
                    f"not multiple values. Use separate list items if needed."
                )
            elif error_type == "missing":
                error_messages.append(
                    f"Field '({field_path} ...)' is REQUIRED and missing. "
                    f"Add this field in parentheses."
                )
            elif input_value is None:
                error_messages.append(
                    f"Field '({field_path})' cannot be empty. "
                    f'Provide a value: ({field_path} "value")'
                )
            else:
                msg = err["msg"]
                error_messages.append(f"Field '({field_path})': {msg}")

        return "Validation errors:\n" + "\n".join(
            f"  - {msg}" for msg in error_messages
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize S-Expression into atoms and parentheses."""
        tokens = []
        i = 0

        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue

            # Handle parentheses
            if text[i] in "()":
                tokens.append(text[i])
                i += 1
                continue

            # Handle quoted strings
            if text[i] == '"':
                j = i + 1
                # Find closing quote
                while j < len(text) and text[j] != '"':
                    if text[j] == "\\" and j + 1 < len(text):
                        j += 2  # Skip escaped character
                    else:
                        j += 1
                if j < len(text):
                    tokens.append(text[i : j + 1])  # Include quotes
                    i = j + 1
                else:
                    raise ValueError("Unclosed quoted string")
                continue

            # Handle atoms (unquoted strings)
            j = i
            while j < len(text) and not text[j].isspace() and text[j] not in '()"':
                j += 1
            if j > i:
                tokens.append(text[i:j])
                i = j
            else:
                i += 1

        return tokens

    def _parse_tokens(
        self, tokens: List[str], pos: int
    ) -> tuple[Union[str, List], int]:
        """Parse tokens into nested lists (S-Expression tree)."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of tokens")

        token = tokens[pos]

        if token == "(":
            # Start of list
            result = []
            pos += 1
            while pos < len(tokens) and tokens[pos] != ")":
                item, pos = self._parse_tokens(tokens, pos)
                result.append(item)
            if pos >= len(tokens):
                raise ValueError("Unmatched opening parenthesis")
            pos += 1  # Skip closing paren
            return result, pos
        elif token == ")":
            raise ValueError("Unexpected closing parenthesis")
        else:
            # Atom (remove quotes if present)
            if token.startswith('"') and token.endswith('"'):
                return token[1:-1], pos + 1
            return token, pos + 1

    def _sexp_to_dict(self, sexp: List) -> Dict[str, Any]:
        """Convert parsed S-Expression to dictionary."""
        if not isinstance(sexp, list) or len(sexp) == 0:
            raise ValueError("Invalid S-Expression structure")

        result = {}

        # Skip first element (object name) and process rest as field pairs
        i = 1
        while i < len(sexp):
            if not isinstance(sexp[i], list) or len(sexp[i]) < 1:
                i += 1
                continue

            field_list = sexp[i]
            field_name = field_list[0]

            if len(field_list) == 1:
                # Empty field
                result[field_name] = None
            elif len(field_list) == 2:
                # Single value or nested object/list
                value = field_list[1]
                if isinstance(value, list):
                    # Check if this looks like a type name (starts with uppercase) = list item
                    if value and isinstance(value[0], str) and value[0][0].isupper():
                        # Treat as a list with one item
                        result[field_name] = [self._sexp_to_dict(value)]
                    else:
                        # Regular nested object
                        result[field_name] = self._sexp_to_dict(value)
                else:
                    result[field_name] = value
            else:
                # Multiple values = list, dict, or nested objects
                values = field_list[1:]

                # Check if this looks like a dict (all 2-element lists with string keys)
                if all(
                    isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
                    for v in values
                ):
                    # Dict representation: ((key1 val1) (key2 val2) ...)
                    dict_result = {}
                    for pair in values:
                        key = pair[0]
                        val = pair[1]
                        if isinstance(val, list):
                            dict_result[key] = self._sexp_to_dict(val)
                        else:
                            dict_result[key] = val
                    result[field_name] = dict_result
                # Check if it's a malformed dict (looks like dict but has pairs with 3 elements)
                # Only treat as malformed dict if keys are lowercase (field names, not types)
                elif all(
                    isinstance(v, list)
                    and len(v) >= 2
                    and isinstance(v[0], str)
                    and v[0][0].islower()
                    for v in values
                ):
                    # Malformed dict - keys have multiple values like (key "val1" "val2")
                    # Try to salvage by taking first value only
                    dict_result = {}
                    for pair in values:
                        key = pair[0]
                        # Take first value, warn about the rest
                        val = pair[1]
                        if isinstance(val, list):
                            dict_result[key] = self._sexp_to_dict(val)
                        else:
                            dict_result[key] = val
                        # Log warning if multiple values
                        if len(pair) > 2:
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.warning(
                                f"S-Exp dict key '{key}' has {len(pair)-1} values, using only first value"
                            )
                    result[field_name] = dict_result
                elif all(isinstance(v, list) for v in values):
                    # List of objects
                    result[field_name] = [self._sexp_to_dict(v) for v in values]
                else:
                    # List of primitives
                    result[field_name] = values

            i += 1

        return result
