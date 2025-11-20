"""
XML format implementation.

Uses XML tags for structured data, potentially more forgiving than JSON.
"""

import re
import xml.etree.ElementTree as ET
from typing import Type, Dict, Any, Optional, get_origin, get_args
from pydantic import BaseModel, ValidationError

from ..base_format import StructuredOutputFormat


class XMLFormat(StructuredOutputFormat):
    """XML format for structured output."""

    def name(self) -> str:
        return "xml"

    @property
    def max_nesting_depth(self) -> Optional[int]:
        return None  # Unlimited

    def generate_schema(self, model: Type[BaseModel]) -> str:
        """Generate XML schema description from Pydantic model."""
        schema_lines = []
        schema_lines.append(f"<{model.__name__}>")
        schema_lines.extend(self._generate_field_schema(model, indent=2))
        schema_lines.append(f"</{model.__name__}>")
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

            # Handle Dict types (map to nested tags in XML)
            origin = get_origin(annotation)
            if origin is dict:
                # Dict fields are represented as nested key-value tags
                lines.append(
                    f"{prefix}<{field_name}>  <!-- Nested key-value tags: {description} -->"
                )
                lines.append(f"{prefix}  <key1>value1</key1>")
                lines.append(f"{prefix}  <key2>value2</key2>")
                lines.append(f"{prefix}</{field_name}>")
            # Handle lists
            elif origin is list:
                args = get_args(annotation)
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    # List of nested objects
                    lines.append(
                        f"{prefix}<{field_name}>  <!-- List of {args[0].__name__} -->"
                    )
                    lines.append(f"{prefix}  <item>")
                    lines.extend(self._generate_field_schema(args[0], indent + 4))
                    lines.append(f"{prefix}  </item>")
                    lines.append(f"{prefix}</{field_name}>")
                else:
                    # List of primitives
                    lines.append(
                        f"{prefix}<{field_name}>  <!-- List: {description} -->"
                    )
                    lines.append(f"{prefix}  <item>value</item>")
                    lines.append(f"{prefix}</{field_name}>")
            # Handle nested objects
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                lines.append(f"{prefix}<{field_name}>  <!-- {description} -->")
                lines.extend(self._generate_field_schema(annotation, indent + 2))
                lines.append(f"{prefix}</{field_name}>")
            # Handle primitives
            else:
                type_name = getattr(annotation, "__name__", str(annotation))
                comment = (
                    f"  <!-- {type_name}: {description} -->" if description else ""
                )
                lines.append(f"{prefix}<{field_name}>value</{field_name}>{comment}")

        return lines

    def build_prompt(self, system_prompt: str, user_input: str, schema_str: str) -> str:
        """Build prompt with XML schema instructions."""
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
            "You must respond with valid XML data that follows this structure:",
            "",
            schema_str,
            "",
            "IMPORTANT:",
            "- Use the exact tag names shown",
            "- Include all required fields",
            "- For lists, wrap each item in <item> tags",
            "- For fields that show nested tags, include the nested key-value tags (not a single text value)",
            "- Do not include any text before or after the XML",
            "- Ensure all tags are properly closed",
        ]

        return "\n".join(prompt_parts)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse XML from response."""
        response_text = response_text.strip()

        # Remove LLM thinking tags (but only if they appear OUTSIDE the main structure)
        # We look for <think> tags, but NOT <reasoning> since that could be a field name
        cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        cleaned_text = cleaned_text.strip()

        # Remove markdown code blocks if present
        if cleaned_text.startswith("```xml") or cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:xml)?\s*\n", "", cleaned_text)
            cleaned_text = re.sub(r"\n```\s*$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        # Extract XML (find outermost tag)
        xml_text = self._extract_xml(cleaned_text)
        if not xml_text:
            raise ValueError("No valid XML found in response")

        # Parse XML
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

        # Convert XML to dict
        return self._xml_to_dict(root)

    def format_error(self, error: ValidationError, model: Type[BaseModel]) -> str:
        """Convert Pydantic errors to clear, XML-friendly messages."""
        error_messages = []

        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            input_value = err.get("input")

            # Format specific error types
            if error_type == "string_type" and isinstance(input_value, list):
                error_messages.append(
                    f"Field '{field_path}' must contain a SINGLE value inside the tag, "
                    f"not multiple values. Create separate <item> tags for multiple values."
                )
            elif error_type == "dict_type" or error_type == "model_type":
                # Nested tags expected, not simple text
                error_messages.append(
                    f"Tag '<{field_path}>' must contain NESTED TAGS with key-value pairs, "
                    f"not a simple text value. Example:\n"
                    f"  <{field_path.split(' -> ')[-1]}>\n"
                    f"    <role1>value1</role1>\n"
                    f"    <role2>value2</role2>\n"
                    f"  </{field_path.split(' -> ')[-1]}>"
                )
            elif error_type == "missing":
                error_messages.append(
                    f"Tag '<{field_path}>' is REQUIRED and missing. "
                    f"You must include this tag in your response."
                )
            elif input_value is None:
                error_messages.append(
                    f"Tag '<{field_path}>' cannot be empty. "
                    f"Provide a value between the opening and closing tags."
                )
            else:
                # Try to translate generic Pydantic messages to XML terminology
                msg = err["msg"]
                # Remove Python-specific terminology
                msg = msg.replace("dictionary", "nested tags")
                msg = msg.replace("Dictionary", "Nested tags")
                msg = msg.replace("dict", "nested tags")
                msg = msg.replace("Dict", "Nested tags")
                error_messages.append(f"Tag '<{field_path}>': {msg}")

        return "Validation errors:\n" + "\n".join(
            f"  - {msg}" for msg in error_messages
        )

    def _extract_xml(self, text: str) -> Optional[str]:
        """Extract XML from text."""
        # Look for first opening tag
        match = re.search(r"<([a-zA-Z_][a-zA-Z0-9_]*)", text)
        if not match:
            return None

        tag_name = match.group(1)
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        start_idx = text.find(start_tag)
        if start_idx == -1:
            return None

        # Find matching end tag (accounting for nested tags)
        depth = 0
        pos = start_idx
        while pos < len(text):
            if text[pos:].startswith(start_tag):
                depth += 1
                pos += len(start_tag)
            elif text[pos:].startswith(end_tag):
                depth -= 1
                if depth == 0:
                    return text[start_idx : pos + len(end_tag)]
                pos += len(end_tag)
            else:
                pos += 1

        return None

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Group children by tag name to handle lists
        children_by_tag = {}
        for child in element:
            tag = child.tag
            if tag not in children_by_tag:
                children_by_tag[tag] = []
            children_by_tag[tag].append(child)

        for tag, children in children_by_tag.items():
            if len(children) == 1:
                child = children[0]
                # Check if it's a list container (has <item> children)
                if len(child) > 0 and all(c.tag == "item" for c in child):
                    # List of items
                    result[tag] = [self._xml_element_value(item) for item in child]
                elif len(child) > 0:
                    # Nested object
                    result[tag] = self._xml_to_dict(child)
                else:
                    # Simple value
                    result[tag] = child.text or ""
            else:
                # Multiple children with same tag = list
                result[tag] = [self._xml_element_value(c) for c in children]

        return result

    def _xml_element_value(self, element: ET.Element) -> Any:
        """Get value from XML element (could be dict or string)."""
        if len(element) > 0:
            return self._xml_to_dict(element)
        return element.text or ""
