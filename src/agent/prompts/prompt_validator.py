"""
Prompt validation utilities.
"""

import logging
import string
from typing import Set

logger = logging.getLogger(__name__)

def extract_template_variables(template: str) -> Set[str]:
    """
    Extract all template variables from a prompt template using Python's string formatter.
    
    Args:
        template: The prompt template string
        
    Returns:
        Set of variable names found in the template
    """
    # Use string.Formatter to parse the template properly
    formatter = string.Formatter()
    variables = set()
    
    # Parse the format string and extract field names
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is not None:
            # Handle both simple names and dotted/indexed access
            # e.g., "user.name" -> "user", "items[0]" -> "items"
            base_name = field_name.split('.')[0].split('[')[0]
            variables.add(base_name)
    
    return variables

def validate_prompt_variables(template: str, provided_variables: Set[str]) -> None:
    """
    Validate that a prompt template only uses provided variables.
    
    Args:
        template: The prompt template string
        provided_variables: Set of variable names that will be provided
        
    Raises:
        ValueError: If template uses variables not in provided_variables
    """
    template_variables = extract_template_variables(template)
    
    # Check for undefined variables
    undefined_variables = template_variables - provided_variables
    if undefined_variables:
        raise ValueError(
            f"Template uses undefined variables: {sorted(undefined_variables)}. "
            f"Available variables: {sorted(provided_variables)}"
        )
    
    # Log unused provided variables
    unused_variables = provided_variables - template_variables
    if unused_variables:
        logger.warning(
            "Provided variables not used in template: %s",
            sorted(unused_variables)
        )