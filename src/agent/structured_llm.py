"""
Structured LLM Calls

Provides a clean abstraction for LLM calls that return structured data,
using Pydantic models for schema generation and automatic validation.
"""

import json
import logging
from enum import Enum
from typing import TypeVar, Type, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from agent.llm import LLM, Message, SupportedModel

logger = logging.getLogger(__name__)


# Import custom format components (lazy import to avoid circular dependencies)
def _get_custom_format_components():
    from agent.custom_format_parser import CustomFormatParser
    from agent.custom_format_schema import CustomFormatSchemaGenerator

    return CustomFormatParser(), CustomFormatSchemaGenerator()


class ResponseFormat(Enum):
    """Supported response formats for structured LLM calls"""

    JSON = "json"
    CUSTOM = "custom"


T = TypeVar("T", bound=BaseModel)


class StructuredLLMError(Exception):
    """Error in structured LLM call"""

    pass


class StructuredLLMClient:
    """Client for making structured LLM calls with automatic validation"""

    def __init__(self, model: SupportedModel, llm: LLM, max_retries: int = 3):
        self.model = model
        self.llm = llm
        self.max_retries = max_retries

    def call(
        self,
        system_prompt: str,
        user_input: str,
        response_model: Type[T],
        caller: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        format: ResponseFormat = ResponseFormat.JSON,
    ) -> T:
        """
        Make a structured LLM call with automatic validation

        Args:
            system_prompt: System instructions for the LLM
            user_input: User input/query
            response_model: Pydantic model class for expected response
            context: Optional context dictionary
            temperature: LLM temperature (lower for more structured output)

        Returns:
            Validated instance of response_model

        Raises:
            StructuredLLMError: If validation fails after retries
        """

        schema_str = build_schema(response_model, format)
        if format == ResponseFormat.JSON:
            full_prompt = self._build_prompt_json(
                system_prompt, user_input, schema_str, context
            )
        elif format == ResponseFormat.CUSTOM:
            full_prompt = self._build_prompt_custom(
                system_prompt, user_input, schema_str, context
            )
        else:
            raise StructuredLLMError(f"Unsupported format: {format}")

        last_error = None

        # Start with the initial messages
        messages = [Message(role="user", content=full_prompt)]

        response_text: str = ""
        json_text: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                # Make LLM call through manager
                response = self.llm.chat_complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    num_predict=4096,
                    caller=caller,
                )

                response_text = response or ""
                if not response_text:
                    raise StructuredLLMError("LLM response is empty")

                # Log raw response for debugging
                logger.debug("Raw LLM response:\n%s", response_text)

                # Parse response based on format
                if format == ResponseFormat.JSON:
                    # Extract and parse JSON
                    json_text = self._extract_json(response_text)
                    if not json_text:
                        raise StructuredLLMError("No valid JSON found in response")

                    # Fix unescaped newlines and control characters in JSON
                    fixed_json = self._fix_json_escaping(json_text)

                    # Parse JSON
                    parsed_data = json.loads(fixed_json)
                elif format == ResponseFormat.CUSTOM:
                    # Parse custom format
                    parser, _ = _get_custom_format_components()
                    parsed_data = parser.parse(response_text, response_model.__name__)
                else:
                    raise StructuredLLMError(f"Unsupported format: {format}")

                # Handle common LLM response pattern where fields are wrapped in "properties"
                if isinstance(parsed_data, dict) and "properties" in parsed_data:
                    # If the model put everything under "properties", unwrap it
                    parsed_data = parsed_data["properties"]

                # Validate with Pydantic model
                validated_result = response_model.model_validate(parsed_data)

                return validated_result

            except (json.JSONDecodeError, ValidationError) as e:
                # Log the full response and error details for debugging
                logger.error(
                    "Structured LLM call failed - attempt %d/%d",
                    attempt + 1,
                    self.max_retries + 1,
                )
                logger.error("Raw LLM response: %s", response_text)
                if format == ResponseFormat.JSON:
                    logger.error(
                        "Extracted JSON text: %s",
                        json_text if "json_text" in locals() else "N/A",
                    )
                logger.error("Error details: %s", str(e))
                logger.error("Error type: %s", type(e).__name__)

                last_error = e

                if attempt < self.max_retries:
                    # Add the failed response to conversation history
                    messages.append(Message(role="assistant", content=response_text))

                    # Add corrective feedback as a new user message
                    error_guidance = self._get_error_guidance(e, schema_str, format)
                    if format == ResponseFormat.JSON:
                        correction_message = f"""That JSON was invalid. {error_guidance}

Please provide a corrected JSON response that matches the required schema exactly. Remember:
- Include ALL required fields
- Use correct data types 
- Respond with ONLY valid JSON, no additional text"""
                    else:
                        correction_message = f"""That response format was invalid. {error_guidance}

Please provide a corrected response that follows the format exactly as specified. Remember:
- Use exact field names as shown
- Include ALL required fields
- Follow the format structure precisely"""

                    messages.append(Message(role="user", content=correction_message))
                    continue
            except Exception as e:
                # Handle other unexpected errors
                logger.error("Unexpected error in structured LLM call: %s", str(e))
                raise StructuredLLMError(f"LLM call failed: {e}")

        # All retries failed
        raise StructuredLLMError(
            f"Failed to get valid response after {self.max_retries + 1} attempts. Last error: {last_error}"
        )

    def _build_prompt_json(
        self,
        system_prompt: str,
        user_input: str,
        schema_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build complete prompt with JSON schema and context"""

        prompt_parts = [
            "You are a helpful AI assistant that provides structured responses.",
            "",
            "TASK:",
            system_prompt,
            "",
        ]

        if context:
            prompt_parts.extend(["CONTEXT:", json.dumps(context, indent=2), ""])

        prompt_parts.extend(
            [
                "INPUT:",
                user_input,
                "",
                "RESPONSE FORMAT:",
                "You must respond with valid JSON data that conforms to this schema:",
                "",
                schema_str,
                "",
                "EXAMPLE:",
                "If the schema requires fields 'name' and 'age', respond with:",
                '{"name": "John", "age": 30}',
                "NOT with the schema definition itself.",
                "",
                "IMPORTANT:",
                "- Create actual data, NOT the schema itself",
                "- Respond ONLY with a JSON object containing the actual values",
                "- Include all required fields with real content",
                "- Follow the field descriptions to generate appropriate values",
                "- Do not include any text before or after the JSON",
                "- Do not return the schema - return data that matches the schema",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_prompt_custom(
        self,
        system_prompt: str,
        user_input: str,
        schema_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build complete prompt with custom format schema and context"""

        prompt_parts = [
            "You are a helpful AI assistant that provides structured responses.",
            "",
            "TASK:",
            system_prompt,
            "",
        ]

        if context:
            prompt_parts.extend(["CONTEXT:", json.dumps(context, indent=2), ""])

        prompt_parts.extend(
            [
                "INPUT:",
                user_input,
                "",
                schema_str,
                "",
                "IMPORTANT:",
                "- Follow the format exactly as shown",
                "- Use the exact field names specified",
                "- All required fields must be included",
                "- Do not include any text outside the specified format",
            ]
        )

        return "\n".join(prompt_parts)

    def _extract_json(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response, cleaning reasoning tags first"""
        response_text = response_text.strip()

        # Remove reasoning tags that could contain misleading JSON-like content
        import re

        cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        cleaned_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", cleaned_text, flags=re.DOTALL
        )
        cleaned_text = cleaned_text.strip()

        # If the response is already just JSON
        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            return cleaned_text

        # Look for JSON within the cleaned response
        if "{" in cleaned_text and "}" in cleaned_text:
            start_idx = cleaned_text.find("{")

            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return cleaned_text[start_idx : i + 1]

        return None

    def _fix_json_escaping(self, json_text: str) -> str:
        """Fix unescaped newlines and control characters in JSON string values"""
        import re

        # Simple approach: find string values and escape them properly
        def fix_string_content(match):
            full_match = match.group(0)
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
        # Matches: "key": "value content here"
        pattern = r'(\s*"[^"]*"\s*:\s*)(")([^"]*?)(?<!\\)"'

        # Apply the fix
        fixed = re.sub(pattern, fix_string_content, json_text, flags=re.DOTALL)

        return fixed

    def _get_error_guidance(
        self, error: Exception, schema_str: str, format: ResponseFormat
    ) -> str:
        """Get specific guidance based on the error type"""

        if isinstance(error, json.JSONDecodeError):
            return f"JSON parsing failed: {error}. Make sure to return valid JSON only."

        elif isinstance(error, ValidationError):
            error_details = []
            for err in error.errors():
                field = " -> ".join(str(x) for x in err["loc"])
                msg = err["msg"]
                error_details.append(f"Field '{field}': {msg}")

            return f"Validation failed:\n" + "\n".join(error_details)

        else:
            return f"Unknown error: {error}"


def build_schema(
    response_model: Type[T],
    format: ResponseFormat,
) -> str:
    if format == ResponseFormat.JSON:
        # Generate JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        return schema_str
    elif format == ResponseFormat.CUSTOM:
        # Generate custom format schema
        _, schema_generator = _get_custom_format_components()
        schema_str = schema_generator.generate_schema_description(response_model)
        return schema_str
    else:
        raise StructuredLLMError(f"Unsupported format: {format}")


# Convenience function for quick structured calls
def structured_llm_call(
    system_prompt: str,
    user_input: str,
    response_model: Type[T],
    model: SupportedModel,
    llm: LLM,
    caller: str,
    context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    format: ResponseFormat = ResponseFormat.JSON,
) -> T:
    """
    Convenience function for making structured LLM calls

    Args:
        system_prompt: System instructions
        user_input: User input
        response_model: Pydantic model for response
        model: Supported model enum
        llm: LLM manager instance
        context: Optional context
        temperature: LLM temperature

    Returns:
        Validated instance of response_model
    """
    structured_client = StructuredLLMClient(model=model, llm=llm)
    return structured_client.call(
        system_prompt, user_input, response_model, caller, context, temperature, format
    )


def direct_structured_llm_call(
    prompt: str,
    response_model: Type[T],
    model: SupportedModel,
    llm: LLM,
    caller: str,
    context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    format: ResponseFormat = ResponseFormat.JSON,
    max_retries: int = 3,
) -> T:
    """
    Direct generation structured LLM call for first-person prompts

    Uses direct generation (llm.generate_complete) instead of chat templates.
    This approach is better suited for first-person consciousness prompts where
    the AI thinks from its own perspective rather than following system/user roles.

    Args:
        prompt: Single first-person prompt
        response_model: Pydantic model for response
        model: Supported model enum
        llm: LLM manager instance
        context: Optional context
        temperature: LLM temperature
        format: Response format (JSON or CUSTOM)
        max_retries: Maximum retry attempts

    Returns:
        Validated instance of response_model

    Raises:
        StructuredLLMError: If validation fails after retries
    """

    # Build the complete prompt with schema and context
    schema_str = build_schema(response_model, format)
    full_prompt = _build_direct_prompt(prompt, schema_str, context, format)

    last_error = None

    for attempt in range(max_retries + 1):
        response_text = ""  # Initialize to avoid unbound variable
        try:
            # Use direct generation instead of chat template
            response_text = llm.generate_complete(
                model=model,
                prompt=full_prompt,
                temperature=temperature,
                num_predict=4096,
                caller=caller,
            )

            if not response_text:
                raise StructuredLLMError("LLM response is empty")

            # Log raw response for debugging
            logger.debug("Raw LLM response (direct generation):\n%s", response_text)

            # Parse response based on format
            if format == ResponseFormat.JSON:
                # Extract and parse JSON
                json_text = _extract_json_from_response(response_text)
                logger.debug(f"DIRECT_DEBUG: Extracted JSON text: {repr(json_text)}")
                if not json_text:
                    raise StructuredLLMError("No valid JSON found in response")

                # Fix unescaped newlines and control characters in JSON
                fixed_json = _fix_json_escaping_standalone(json_text)
                logger.debug(f"DIRECT_DEBUG: Fixed JSON: {repr(fixed_json)}")

                # Parse JSON
                parsed_data = json.loads(fixed_json)
                logger.debug(f"DIRECT_DEBUG: Parsed data: {parsed_data}")
                logger.debug(f"DIRECT_DEBUG: Parsed data type: {type(parsed_data)}")
                if isinstance(parsed_data, dict):
                    logger.debug(
                        f"DIRECT_DEBUG: Parsed data keys: {list(parsed_data.keys())}"
                    )
            elif format == ResponseFormat.CUSTOM:
                # Parse custom format
                parser, _ = _get_custom_format_components()
                parsed_data = parser.parse(response_text, response_model.__name__)
            else:
                raise StructuredLLMError(f"Unsupported format: {format}")

            # Handle common LLM response pattern where fields are wrapped in "properties"
            if isinstance(parsed_data, dict) and "properties" in parsed_data:
                logger.debug(
                    f"DIRECT_DEBUG: Found properties key, unwrapping from: {parsed_data}"
                )
                # If the model put everything under "properties", unwrap it
                parsed_data = parsed_data["properties"]
                logger.debug(
                    f"DIRECT_DEBUG: After properties unwrapping: {parsed_data}"
                )

            # Validate with Pydantic model
            logger.debug(f"DIRECT_DEBUG: Input to validation: {parsed_data}")
            logger.debug(f"DIRECT_DEBUG: Input type to validation: {type(parsed_data)}")
            validated_result = response_model.model_validate(parsed_data)

            return validated_result

        except (json.JSONDecodeError, ValidationError) as e:
            # Log the full response and error details for debugging
            logger.error(
                "Direct structured LLM call failed - attempt %d/%d",
                attempt + 1,
                max_retries + 1,
            )
            logger.error("Raw LLM response: %s", response_text)
            logger.error("Error details: %s", str(e))
            logger.error("Error type: %s", type(e).__name__)

            last_error = e

            if attempt < max_retries:
                # For direct generation, we need to modify the prompt for retry
                error_guidance = _get_error_guidance_standalone(e, schema_str, format)
                if format == ResponseFormat.JSON:
                    correction_text = f"""

PREVIOUS ATTEMPT FAILED: {error_guidance}

Please provide a corrected JSON response that matches the required schema exactly. Remember:
- Include ALL required fields
- Use correct data types 
- Respond with ONLY valid JSON, no additional text"""
                else:
                    correction_text = f"""

PREVIOUS ATTEMPT FAILED: {error_guidance}

Please provide a corrected response that follows the format exactly as specified. Remember:
- Use exact field names as shown
- Include ALL required fields
- Follow the format structure precisely"""

                full_prompt += correction_text
                continue

        except Exception as e:
            # Handle other unexpected errors
            logger.error("Unexpected error in direct structured LLM call: %s", str(e))
            raise StructuredLLMError(f"Direct LLM call failed: {e}")

    # All retries failed
    raise StructuredLLMError(
        f"Failed to get valid response after {max_retries + 1} attempts. Last error: {last_error}"
    )


def _build_direct_prompt(
    prompt: str,
    schema_str: str,
    context: Optional[Dict[str, Any]] = None,
    format: ResponseFormat = ResponseFormat.JSON,
) -> str:
    """Build complete direct generation prompt with schema and context"""

    prompt_parts = [prompt, ""]

    if context:
        prompt_parts.extend(["CONTEXT:", json.dumps(context, indent=2), ""])

    if format == ResponseFormat.JSON:
        prompt_parts.extend(
            [
                "I need to provide my response in JSON format that conforms to this schema:",
                "",
                schema_str,
                "",
                "EXAMPLE:",
                "If the schema requires fields 'name' and 'age', I respond with:",
                '{"name": "John", "age": 30}',
                "NOT with the schema definition itself.",
                "",
                "IMPORTANT:",
                "- I'll create actual data, NOT the schema itself",
                "- I'll respond ONLY with a JSON object containing the actual values",
                "- I'll include all required fields with real content",
                "- I'll follow the field descriptions to generate appropriate values",
                "- I won't include any text before or after the JSON",
                "- I won't return the schema - I'll return data that matches the schema",
                "",
                "Here is my response in JSON format:",
            ]
        )
    elif format == ResponseFormat.CUSTOM:
        prompt_parts.extend(
            [
                schema_str,
                "",
                "IMPORTANT:",
                "- I'll follow the format exactly as shown",
                "- I'll use the exact field names specified",
                "- I'll include all required fields",
                "- I won't include any text outside the specified format",
                "",
                "Here is my response:",
            ]
        )

    return "\n".join(prompt_parts)


def _extract_json_from_response(response_text: str) -> Optional[str]:
    """Extract JSON from direct generation response, cleaning reasoning tags first"""
    response_text = response_text.strip()

    # Remove reasoning tags that could contain misleading JSON-like content
    import re

    cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    cleaned_text = re.sub(
        r"<reasoning>.*?</reasoning>", "", cleaned_text, flags=re.DOTALL
    )
    cleaned_text = cleaned_text.strip()

    # If the response is already just JSON
    if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
        return cleaned_text

    # Look for JSON within the cleaned response
    if "{" in cleaned_text and "}" in cleaned_text:
        start_idx = cleaned_text.find("{")

        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(cleaned_text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return cleaned_text[start_idx : i + 1]

    return None


def _fix_json_escaping_standalone(json_text: str) -> str:
    """Fix unescaped newlines and control characters in JSON string values"""
    import re

    # Simple approach: find string values and escape them properly
    def fix_string_content(match):
        full_match = match.group(0)
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
    # Matches: "key": "value content here"
    pattern = r'(\s*"[^"]*"\s*:\s*)(")([^"]*?)(?<!\\)"'

    # Apply the fix
    fixed = re.sub(pattern, fix_string_content, json_text, flags=re.DOTALL)

    return fixed


def _get_error_guidance_standalone(
    error: Exception, schema_str: str, format: ResponseFormat
) -> str:
    """Get specific guidance based on the error type for direct generation"""

    if isinstance(error, json.JSONDecodeError):
        return f"JSON parsing failed: {error}. Make sure to return valid JSON only."

    elif isinstance(error, ValidationError):
        error_details = []
        for err in error.errors():
            field = " -> ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            error_details.append(f"Field '{field}': {msg}")

        return f"Validation failed:\n" + "\n".join(error_details)

    else:
        return f"Unknown error: {error}"


def main():
    """Test the structured LLM system"""
    print("=== STRUCTURED LLM SYSTEM TEST ===")

    # Define test models
    from pydantic import BaseModel, Field
    from typing import List
    from agent.llm import create_llm, SupportedModel

    class PreferenceType(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"
        UNKNOWN = "unknown"

    class UserPreference(BaseModel):
        description: str = Field(
            description="Clear description of what the user values or wants to avoid"
        )
        preference_type: PreferenceType = Field(
            description="Specifies whether this is a positive, negative, or neutral preference"
        )
        confidence: float = Field(
            description="Confidence level from 0.0 to 1.0", ge=0.0, le=1.0
        )
        evidence: List[str] = Field(
            description="Specific quotes or examples from the user feedback that support this preference"
        )

    class PreferenceAnalysis(BaseModel):
        preferences: List[UserPreference] = Field(
            description="List of extracted user preferences"
        )
        core_insights: List[str] = Field(
            description="Broader themes about what the user fundamentally values"
        )
        overall_confidence: float = Field(
            description="Overall confidence in the analysis", ge=0.0, le=1.0
        )

    # Test structured call
    print("\n🧪 Testing preference extraction...")

    system_prompt = """You are an expert at understanding user preferences from feedback. 
    Extract the underlying preferences and values from user feedback about conversations."""

    user_input = """This character felt really engaging and had great personality depth. 
    I loved how they stayed consistent throughout the conversation and used tools appropriately. 
    However, the dialogue was a bit repetitive and could have been more varied."""

    context = {
        "domain": "conversation",
        "scenario": "vampire roleplay",
        "conversation_length": 8,
    }

    llm = create_llm()

    try:
        result = structured_llm_call(
            system_prompt=system_prompt,
            user_input=user_input,
            response_model=PreferenceAnalysis,
            context=context,
            model=SupportedModel.MISTRAL_SMALL,
            llm=llm,
            caller="structured_llm_test",
        )

        print(f"✅ Successfully extracted {len(result.preferences)} preferences")
        print(f"Overall confidence: {result.overall_confidence:.2f}")

        print("\nExtracted preferences:")
        for i, pref in enumerate(result.preferences, 1):
            print(f"  {i}. {pref.preference_type}: {pref.description}")
            print(f"     Confidence: {pref.confidence:.2f}")
            print(f"     Evidence: {pref.evidence}")

        print(f"\nCore insights:")
        for insight in result.core_insights:
            print(f"  • {insight}")

    except StructuredLLMError as e:
        print(f"❌ Structured LLM call failed: {e}")

    # Test schema generation
    print(f"\n📋 Generated JSON Schema:")
    schema = build_schema(
        response_model=PreferenceAnalysis,
        format=ResponseFormat.CUSTOM,
    )
    print(schema)

    print("\n✅ Structured LLM system test complete!")


if __name__ == "__main__":
    main()
