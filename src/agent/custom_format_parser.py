"""
Custom Text Format Parser for Essay-Like Content

Parses a custom text format designed for LLM-generated structured responses
that contain essay-like content with automatic indentation normalization.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class CustomFormatError(Exception):
    """Error parsing custom format"""

    pass


class CustomFormatParser:
    """Parser for custom text format with essay-like content support"""

    def __init__(self):
        # Regex patterns for parsing (be more tolerant of missing > characters)
        self.field_pattern = re.compile(
            r"<<<START\s+([A-Z_][A-Z0-9_]*)\s*>*>(.*?)<<<END\s+\1\s*>*>",
            re.DOTALL | re.IGNORECASE,
        )
        # Tolerant pattern that matches various malformed ITEM markers
        self.item_pattern = re.compile(r"<{1,3}ITEM>{1,3}", re.IGNORECASE)

    def parse(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Parse custom format text into a dictionary

        Args:
            text: Raw text in custom format
            model_name: Model name to unwrap if used as root wrapper

        Returns:
            Dictionary with parsed fields

        Raises:
            CustomFormatError: If parsing fails
        """
        try:
            # Validate format first
            validation_errors = self.validate_format(text)
            if validation_errors:
                raise CustomFormatError(
                    f"Format validation failed: {'; '.join(validation_errors)}"
                )

            result = {}

            # Find all field matches
            for match in self.field_pattern.finditer(text):
                field_name = match.group(1)  # Keep original case
                field_content = match.group(2)

                # Parse the field content
                parsed_value = self._parse_field_content(field_content, field_name)
                result[field_name] = parsed_value

            # Handle model wrapper - if we have exactly one field matching the model name, unwrap it
            if len(result) == 1:
                single_field_name = list(result.keys())[0]
                if single_field_name.lower() == model_name.lower() and isinstance(
                    result[single_field_name], dict
                ):
                    logger.debug(
                        f"Unwrapping model wrapper '{single_field_name}' (matches '{model_name}')"
                    )
                    return result[single_field_name]

            return result

        except Exception as e:
            logger.error(f"Failed to parse custom format: {e}")
            logger.error(f"Input text: {text}")
            raise CustomFormatError(f"Parsing failed: {e}")

    def _parse_field_content(
        self, content: str, field_name: str
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Parse individual field content"""

        # Check if this contains nested fields first (higher priority than arrays)
        nested_matches = list(self.field_pattern.finditer(content))
        if nested_matches:
            return self._parse_nested_content(content)

        # Check if this is an array (contains <<<ITEM>>> markers)
        if self.item_pattern.search(content):
            return self._parse_array_content(content)

        # Simple field - normalize indentation and return
        return self._normalize_content(content)

    def _parse_array_content(self, content: str) -> List[str]:
        """Parse array content separated by <<<ITEM>>> markers"""

        # Split on <<<ITEM>>> markers
        items = self.item_pattern.split(content)

        # First item is before the first <<<ITEM>>>, usually empty
        if items and not items[0].strip():
            items = items[1:]

        # Normalize each item
        normalized_items = []
        for item in items:
            normalized = self._normalize_content(item)
            if normalized:  # Skip empty items
                normalized_items.append(normalized)

        return normalized_items

    def _parse_nested_content(self, content: str) -> Dict[str, Any]:
        """Parse nested object content"""

        result = {}

        # Find all nested fields
        for match in self.field_pattern.finditer(content):
            field_name = match.group(1)  # Keep original case
            field_content = match.group(2)

            # Recursively parse nested content
            parsed_value = self._parse_field_content(field_content, field_name)
            result[field_name] = parsed_value

        return result

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content indentation by finding minimum indent and setting it to 0
        """
        if not content:
            return ""

        lines = content.strip("\n").split("\n")
        if not lines:
            return ""

        # Find minimum indentation of non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return content.strip()

        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        # Remove common leading indentation from all lines
        normalized_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                normalized_lines.append(
                    line[min_indent:] if len(line) >= min_indent else line
                )
            else:  # Empty line - preserve as empty
                normalized_lines.append("")

        # Join and strip trailing whitespace
        result = "\n".join(normalized_lines).rstrip()
        return result

    def validate_format(self, text: str) -> List[str]:
        """
        Validate format and return list of error messages

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for balanced START/END tags (be more tolerant of missing > characters)
        start_tags = re.findall(
            r"<<<START\s+([A-Z_][A-Z0-9_]*)\s*>*>", text, re.IGNORECASE
        )
        end_tags = re.findall(r"<<<END\s+([A-Z_][A-Z0-9_]*)\s*>*>", text, re.IGNORECASE)

        # Convert to lowercase for comparison
        start_tags = [tag.lower() for tag in start_tags]
        end_tags = [tag.lower() for tag in end_tags]

        # Check counts
        if len(start_tags) != len(end_tags):
            errors.append(
                f"Unmatched START/END tags: {len(start_tags)} START tags, {len(end_tags)} END tags"
            )

        # Check for missing END tags for each START tag
        for tag in start_tags:
            if tag not in end_tags:
                errors.append(f"Missing <<<END {tag.upper()}>>> tag")

        # Check for extra END tags
        for tag in end_tags:
            if tag not in start_tags:
                errors.append(
                    f"Extra <<<END {tag.upper()}>>> tag without matching START"
                )

        # Check for properly nested tags (be more tolerant of missing > characters)
        tag_stack = []
        for match in re.finditer(
            r"<<<(START|END)\s+([A-Z_][A-Z0-9_]*)\s*>*", text, re.IGNORECASE
        ):
            tag_type = match.group(1).upper()
            tag_name = match.group(2).lower()

            if tag_type == "START":
                tag_stack.append(tag_name)
            elif tag_type == "END":
                if not tag_stack:
                    errors.append(
                        f"<<<END {tag_name.upper()}>>> without matching START tag"
                    )
                elif tag_stack[-1] != tag_name:
                    expected = tag_stack[-1]
                    errors.append(
                        f"Mismatched tags: expected <<<END {expected.upper()}>>>, got <<<END {tag_name.upper()}>>>"
                    )
                else:
                    tag_stack.pop()

        if tag_stack:
            for tag in tag_stack:
                errors.append(f"Unclosed <<<START {tag.upper()}>>> tag")

        return errors

    def generate_helpful_error_message(self, text: str, errors: List[str]) -> str:
        """
        Generate a helpful error message for the LLM to understand what went wrong
        """
        if not errors:
            return "Format is valid"

        message_parts = ["Format validation failed. Please fix these issues:", ""]

        for i, error in enumerate(errors, 1):
            message_parts.append(f"{i}. {error}")

        message_parts.extend(
            [
                "",
                "Remember the format rules:",
                "- Each field must have matching <<<START FIELD_NAME>>> and <<<END FIELD_NAME>>> tags",
                "- Field names must be UPPERCASE with underscores",
                "- Tags must be properly nested",
                "- For arrays, use <<<ITEM>>> to separate items within the field",
                "",
                "Example:",
                "<<<START IMPROVED_PROMPT>>>",
                "Your content here",
                "<<<END IMPROVED_PROMPT>>>",
            ]
        )

        return "\n".join(message_parts)


def main():
    """Test the custom format parser"""
    print("=== CUSTOM FORMAT PARSER TEST ===")

    # Test basic parsing
    test_text = """
<<<START IMPROVED_PROMPT>>>
You are a helpful assistant with enhanced personality.

When responding:
  1. Be friendly
  2. Be clear
  3. Be helpful
<<<END IMPROVED_PROMPT>>>

<<<START RATIONALE>>>
The original prompt lacked personality and structure.
Added clear guidelines for response behavior.
<<<END RATIONALE>>>

<<<START TARGETED_ISSUES>>>
  <<<ITEM>>>
  Lack of personality in responses
  <<<ITEM>>>
  No clear structure for interactions
  <<<ITEM>>>
  Missing behavioral guidelines
<<<END TARGETED_ISSUES>>>

<<<START CONFIDENCE>>>
0.9
<<<END CONFIDENCE>>>
"""

    parser = CustomFormatParser()

    try:
        result = parser.parse(test_text, "PromptMutation")
        print("✅ Parsing successful!")
        print(f"Fields found: {list(result.keys())}")
        print(f"Improved prompt length: {len(result.get('improved_prompt', ''))}")
        print(f"Number of issues: {len(result.get('targeted_issues', []))}")
        print(f"Confidence: {result.get('confidence')}")

        # Show first issue
        if result.get("targeted_issues"):
            print(f"First issue: {result['targeted_issues'][0]}")

    except CustomFormatError as e:
        print(f"❌ Parsing failed: {e}")

    print("\n✅ Custom format parser test complete!")


if __name__ == "__main__":
    main()
