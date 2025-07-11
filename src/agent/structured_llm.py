"""
Structured LLM Calls

Provides a clean abstraction for LLM calls that return structured data,
using Pydantic models for schema generation and automatic validation.
"""

import json
from typing import TypeVar, Type, Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError

from agent.llm import LLM, SupportedModel, Message, create_llm


T = TypeVar("T", bound=BaseModel)


class StructuredLLMError(Exception):
    """Error in structured LLM call"""

    pass


class StructuredLLMClient:
    """Client for making structured LLM calls with automatic validation"""

    def __init__(self, model: SupportedModel, llm: LLM, max_retries: int = 2):
        self.model = model
        self.llm = llm
        self.max_retries = max_retries

    def call(
        self,
        system_prompt: str,
        user_input: str,
        response_model: Type[T],
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
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

        # Generate JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Build complete prompt with schema
        full_prompt = self._build_prompt(system_prompt, user_input, schema_str, context)

        last_error = None

        # Start with the initial messages
        messages = [Message(role="user", content=full_prompt)]

        for attempt in range(self.max_retries + 1):
            try:
                # Make LLM call through manager
                response = self.llm.chat_complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    num_predict=4096,
                )

                response_text = response
                assert response_text, "LLM response is empty"

                # Extract and parse JSON
                json_text = self._extract_json(response_text)
                if not json_text:
                    raise StructuredLLMError("No valid JSON found in response")

                try:
                    # Parse JSON
                    parsed_data = json.loads(json_text)

                    # Validate with Pydantic model
                    validated_result = response_model.model_validate(parsed_data)

                    return validated_result

                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = e

                    if attempt < self.max_retries:
                        # Add the failed response to conversation history
                        messages.append(
                            Message(role="assistant", content=response_text)
                        )

                        # Add corrective feedback as a new user message
                        error_guidance = self._get_error_guidance(e, schema_str)
                        correction_message = f"""That JSON was invalid. {error_guidance}

    Please provide a corrected JSON response that matches the required schema exactly. Remember:
    - Include ALL required fields
    - Use correct data types 
    - Respond with ONLY valid JSON, no additional text"""

                        messages.append(
                            Message(role="user", content=correction_message)
                        )
                        continue

            except Exception as e:
                raise StructuredLLMError(f"LLM call failed: {e}")

        # All retries failed
        raise StructuredLLMError(
            f"Failed to get valid response after {self.max_retries + 1} attempts. Last error: {last_error}"
        )

    def _build_prompt(
        self,
        system_prompt: str,
        user_input: str,
        schema_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build complete prompt with schema and context"""

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
                "You must respond with valid JSON that matches this exact schema:",
                "",
                schema_str,
                "",
                "IMPORTANT:",
                "- Respond ONLY with valid JSON",
                "- Include all required fields",
                "- Follow the field descriptions in the schema",
                "- Do not include any text before or after the JSON",
                "- Ensure all field types match the schema",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_retry_prompt(
        self,
        system_prompt: str,
        user_input: str,
        schema_str: str,
        context: Optional[Dict[str, Any]],
        error_guidance: str,
    ) -> str:
        """Build retry prompt with error-specific guidance"""

        base_prompt = self._build_prompt(system_prompt, user_input, schema_str, context)

        retry_guidance = f"""

PREVIOUS ATTEMPT FAILED:
{error_guidance}

Please try again, being extra careful to:
- Return only valid JSON
- Match the exact schema structure
- Include all required fields
- Use correct data types"""

        return base_prompt + retry_guidance

    def _extract_json(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response"""
        response_text = response_text.strip()

        # If the response is already just JSON
        if response_text.startswith("{") and response_text.endswith("}"):
            return response_text

        # Look for JSON within the response
        if "{" in response_text and "}" in response_text:
            start_idx = response_text.find("{")

            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return response_text[start_idx : i + 1]

        return None

    def _get_error_guidance(self, error: Exception, schema_str: str) -> str:
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


# Convenience function for quick structured calls
def structured_llm_call(
    system_prompt: str,
    user_input: str,
    response_model: Type[T],
    model: SupportedModel,
    llm: LLM,
    context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
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
        system_prompt, user_input, response_model, context, temperature
    )


def main():
    """Test the structured LLM system"""
    print("=== STRUCTURED LLM SYSTEM TEST ===")

    # Define test models
    from pydantic import BaseModel, Field
    from typing import List

    class UserPreference(BaseModel):
        description: str = Field(
            description="Clear description of what the user values or wants to avoid"
        )
        preference_type: str = Field(
            description="Either 'positive' (what they like) or 'negative' (what they dislike)"
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
    schema = PreferenceAnalysis.model_json_schema()
    print(json.dumps(schema, indent=2)[:500] + "...")

    print("\n✅ Structured LLM system test complete!")


if __name__ == "__main__":
    main()
