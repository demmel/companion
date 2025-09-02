"""
Custom Format Schema Generator

Converts Pydantic models to custom format schema descriptions that help LLMs
understand the expected structure for essay-like content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Type, Dict, Any, List, Union, Optional, Set
from pydantic import BaseModel


@dataclass
class StringTypeInfo:
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    type: str = "string"
    description: str = "string"


@dataclass
class EnumTypeInfo:
    allowed_values: List[str]
    type: str = "enum"
    description: str = "enum"


@dataclass
class NumberTypeInfo:
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    type: str = "number"
    description: str = "number"


@dataclass
class IntegerTypeInfo:
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    type: str = "integer"
    description: str = "integer"


@dataclass
class BooleanTypeInfo:
    type: str = "boolean"
    description: str = "boolean"


@dataclass
class ArrayTypeInfo:
    description: str
    items_type: TypeInfo
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    type: str = "array"


@dataclass
class ObjectTypeInfo:
    description: str
    referenced_type: str  # Name in $defs
    type: str = "object"


@dataclass
class UnionTypeInfo:
    description: str
    options: List["TypeInfo"]
    type: str = "union"


TypeInfo = Union[
    StringTypeInfo,
    EnumTypeInfo,
    NumberTypeInfo,
    IntegerTypeInfo,
    BooleanTypeInfo,
    ArrayTypeInfo,
    ObjectTypeInfo,
    UnionTypeInfo,
]


@dataclass
class TypeDefinition:
    name: str
    kind: str  # "enum", "object", etc.
    properties: Dict[str, TypeInfo]
    required_fields: Set[str]
    constraints: List[str]


@dataclass
class ResolvedSchema:
    root_fields: Dict[str, TypeInfo]
    type_definitions: Dict[str, TypeDefinition]
    required_fields: Set[str]


class CustomFormatSchemaGenerator:
    """Generates custom format schema descriptions from Pydantic models"""

    def __init__(self):
        pass

    def generate_schema_description(self, model: Type[BaseModel]) -> str:
        """
        Generate a human-readable schema description for the custom format

        Args:
            model: Pydantic model class

        Returns:
            String description of the expected format
        """
        schema = model.model_json_schema()

        description_parts = [
            f"RESPONSE FORMAT: {model.__name__}",
            "",
            "ROOT RESPONSE STRUCTURE:",
            f"You must respond with a {model.__name__} object containing these fields:",
        ]

        # Add root-level field descriptions
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))
        array_field_example = None
        referenced_types = set()

        for field_name, field_info in properties.items():
            type_info = self._extract_type_info(field_info, schema)
            is_required = field_name in required_fields
            field_desc = field_info.get("description", "")

            # Build field line
            required_text = "required" if is_required else "optional"
            field_line = f"- {field_name} ({type_info.description}, {required_text})"
            if field_desc:
                field_line += f": {field_desc}"

            description_parts.append(field_line)

            # Add constraints if present
            constraints = self._format_constraints_for_type_info(type_info)
            if constraints:
                description_parts.append(f"  Constraints: {constraints}")

            # Track referenced types for later definition
            if isinstance(type_info, ObjectTypeInfo):
                referenced_types.add(type_info.referenced_type)
            elif isinstance(type_info, ArrayTypeInfo) and isinstance(
                type_info.items_type, ObjectTypeInfo
            ):
                referenced_types.add(type_info.items_type.referenced_type)

            # Save first array field for example
            if isinstance(type_info, ArrayTypeInfo) and array_field_example is None:
                array_field_example = field_name

        # Add defined types section if we have any
        if referenced_types or schema.get("$defs"):
            description_parts.extend(["", "DEFINED TYPES:", ""])

            # Add types from $defs
            defs = schema.get("$defs", {})
            for type_name in sorted(referenced_types.union(defs.keys())):
                if type_name in defs:
                    type_def = defs[type_name]
                    type_description = self._generate_type_description(
                        type_name, type_def, schema
                    )
                    description_parts.extend(type_description)
                    description_parts.append("")

        # Add syntax rules
        description_parts.extend(
            [
                "SYNTAX RULES:",
                "- All system markers use exactly 3 angle brackets: <<<MARKER>>>",
                "- Field structure: <<<START field_name>>> content <<<END field_name>>>",
                "- Array items: <<<ITEM>>> (separates each array element)",
                "- IMPORTANT: Use FIELD NAMES for tags, NOT type names or class names",
                "- Example: For field 'reason' use <<<START reason>>>value<<<END reason>>>",
                "- NEVER use type names like <<<START InterruptionReason>>>value<<<END InterruptionReason>>>",
                "- Field names must match the schema exactly (case-sensitive)",
                "- Content can span multiple lines",
                "",
                "NESTED OBJECTS:",
                "- Wrap the entire object with the field name",
                "- Include all sub-fields inside the wrapper",
                "- Example for nested object:",
                "  <<<START address>>>",
                "  <<<START street>>>123 Main St<<<END street>>>",
                "  <<<START city>>>Springfield<<<END city>>>",
                "  <<<END address>>>",
                "",
                "ARRAYS:",
                "- Use <<<ITEM>>> separators for each element",
                "- For arrays of objects, nest the object structure inside each <<<ITEM>>>",
                "- IMPORTANT: Only use <<<ITEM>>> for arrays, NOT for single values or enums",
                "",
                "ENUMS (single values):",
                "- Example: <<<START reason>>>low_confidence<<<END reason>>>",
                "- Never use <<<ITEM>>> for enums: <<<START reason>>><<<ITEM>>>low_confidence<<<END reason>>> is WRONG",
            ]
        )

        # Add array example if we have array fields
        if array_field_example:
            # Check if it's an array of objects or primitives
            array_type_info = None
            for field_name, field_info in properties.items():
                if field_name == array_field_example:
                    array_type_info = self._extract_type_info(field_info, schema)
                    break

            if isinstance(array_type_info, ArrayTypeInfo):
                if isinstance(array_type_info.items_type, ObjectTypeInfo):
                    # Array of objects example
                    description_parts.extend(
                        [
                            "- Example array of objects:",
                            f"  <<<START {array_field_example}>>>",
                            "  <<<ITEM>>>",
                            "  <<<START field1>>>value1<<<END field1>>>",
                            "  <<<START field2>>>value2<<<END field2>>>",
                            "  <<<ITEM>>>",
                            "  <<<START field1>>>value3<<<END field1>>>",
                            "  <<<START field2>>>value4<<<END field2>>>",
                            f"  <<<END {array_field_example}>>>",
                        ]
                    )
                else:
                    # Array of primitives example
                    description_parts.extend(
                        [
                            "- Example array of simple values:",
                            f"  <<<START {array_field_example}>>>",
                            "  <<<ITEM>>>",
                            "  First item",
                            "  <<<ITEM>>>",
                            "  Second item",
                            f"  <<<END {array_field_example}>>>",
                        ]
                    )

        return "\n".join(description_parts)

    def _extract_type_info(
        self,
        field_info: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> TypeInfo:
        """Extract type information handling Pydantic's complex type representations"""

        # Handle anyOf (common for Optional types)
        if "anyOf" in field_info:
            any_of = field_info["anyOf"]
            # Filter out null types to get the actual type
            non_null_types = [t for t in any_of if t.get("type") != "null"]
            if len(non_null_types) == 1:
                actual_type = non_null_types[0]
                return self._extract_type_info(actual_type, schema)
            else:
                # Handle complex unions
                options = []
                for t in non_null_types:
                    type_info = self._extract_type_info(t, schema)
                    options.append(type_info)
                descriptions = [opt.description for opt in options]
                return UnionTypeInfo(
                    description=f"union of {', '.join(descriptions)}", options=options
                )

        if "$ref" in field_info:
            # Handle references to other models
            ref_path = field_info["$ref"]
            ref_name = ref_path.split("/")[-1]
            return ObjectTypeInfo(
                description=f"{ref_name} object", referenced_type=ref_name
            )

        # Handle regular types
        field_type = field_info.get("type", "unknown")

        if field_type == "string":
            if "enum" in field_info:
                return EnumTypeInfo(allowed_values=[str(v) for v in field_info["enum"]])
            else:
                return StringTypeInfo(
                    min_length=field_info.get("minLength"),
                    max_length=field_info.get("maxLength"),
                    pattern=field_info.get("pattern"),
                )
        elif field_type == "number":
            return NumberTypeInfo(
                minimum=field_info.get("minimum"), maximum=field_info.get("maximum")
            )
        elif field_type == "integer":
            return IntegerTypeInfo(
                minimum=field_info.get("minimum"), maximum=field_info.get("maximum")
            )
        elif field_type == "boolean":
            return BooleanTypeInfo()
        elif field_type == "array":
            items_info = field_info.get("items", {})
            items_type = self._extract_type_info(items_info, schema)

            # Build description based on items type
            if isinstance(items_type, ObjectTypeInfo):
                description = f"array of {items_type.referenced_type} objects"
            else:
                description = f"array of {items_type.description}s"

            return ArrayTypeInfo(
                description=description,
                items_type=items_type,
                min_items=field_info.get("minItems"),
                max_items=field_info.get("maxItems"),
            )
        elif field_type == "object":
            return ObjectTypeInfo(description="object", referenced_type="unknown")
        else:
            # Fallback to string for unknown types
            return StringTypeInfo(description=field_type)

    def _generate_type_description(
        self, type_name: str, type_def: Dict[str, Any], schema: Dict[str, Any]
    ) -> List[str]:
        """Generate description for a type definition"""
        description_lines = []

        # Type header with kind
        type_kind = self._get_type_kind(type_def)
        description_lines.append(f"{type_name} ({type_kind}):")

        if type_kind == "enum":
            # Handle enum types
            constraints = self._format_constraints_for_type_info(
                self._extract_type_info(type_def, schema)
            )
            if constraints:
                description_lines.append(f"- {constraints}")
        elif type_kind == "object":
            # Handle object types
            properties = type_def.get("properties", {})
            required_fields = set(type_def.get("required", []))

            for field_name, field_info in properties.items():
                type_info = self._extract_type_info(field_info, schema)
                is_required = field_name in required_fields
                field_desc = field_info.get("description", "")

                required_text = "required" if is_required else "optional"
                field_line = (
                    f"- {field_name} ({type_info.description}, {required_text})"
                )
                if field_desc:
                    field_line += f": {field_desc}"

                description_lines.append(field_line)

                # Add constraints if present
                constraints = self._format_constraints_for_type_info(type_info)
                if constraints:
                    description_lines.append(f"  Constraints: {constraints}")

        return description_lines

    def _get_type_kind(self, type_def: Dict[str, Any]) -> str:
        """Determine the kind of type (enum, object, etc.)"""
        if "enum" in type_def:
            return "enum"
        elif type_def.get("type") == "object" or "properties" in type_def:
            return "object"
        elif type_def.get("type") == "array":
            return "array"
        elif type_def.get("type") == "string":
            return "string"
        else:
            return type_def.get("type", "unknown")

    def _format_constraints_for_type_info(self, type_info: TypeInfo) -> str:
        """Format constraints for a TypeInfo object"""
        constraints = []

        if isinstance(type_info, StringTypeInfo):
            if type_info.min_length is not None:
                constraints.append(f"min length: {type_info.min_length}")
            if type_info.max_length is not None:
                constraints.append(f"max length: {type_info.max_length}")
            if type_info.pattern is not None:
                constraints.append(f"pattern: {type_info.pattern}")

        elif isinstance(type_info, EnumTypeInfo):
            enum_values = ", ".join(type_info.allowed_values)
            constraints.append(
                f"SINGLE VALUE from: {enum_values} (NOT an array - do not use <<<ITEM>>>)"
            )

        elif isinstance(type_info, NumberTypeInfo):
            if type_info.minimum is not None:
                constraints.append(f"minimum: {type_info.minimum}")
            if type_info.maximum is not None:
                constraints.append(f"maximum: {type_info.maximum}")

        elif isinstance(type_info, IntegerTypeInfo):
            if type_info.minimum is not None:
                constraints.append(f"minimum: {type_info.minimum}")
            if type_info.maximum is not None:
                constraints.append(f"maximum: {type_info.maximum}")

        elif isinstance(type_info, ArrayTypeInfo):
            if type_info.min_items is not None:
                constraints.append(f"min items: {type_info.min_items}")
            if type_info.max_items is not None:
                constraints.append(f"max items: {type_info.max_items}")

        return "; ".join(constraints) if constraints else ""

    def _get_default_constraints_for_type_info(self, type_info: TypeInfo) -> str:
        """Get reasonable default constraint descriptions for types without explicit constraints"""
        if isinstance(type_info, StringTypeInfo):
            return "Any text, can span multiple lines"
        elif isinstance(type_info, (NumberTypeInfo, IntegerTypeInfo)):
            return "Any numeric value"
        elif isinstance(type_info, BooleanTypeInfo):
            return "true or false"
        elif isinstance(type_info, ArrayTypeInfo):
            return "One or more items"
        elif isinstance(type_info, ObjectTypeInfo):
            return "Nested fields"
        elif isinstance(type_info, EnumTypeInfo):
            return f"One of: {', '.join(type_info.allowed_values)}"
        else:
            return "Any value"
