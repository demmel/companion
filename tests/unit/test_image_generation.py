"""
Test suite for image generation tool
"""

import pytest

from agent.types import ToolCallError, ToolCallSuccess, ToolResult
from agent.tools.image_generation_tools import ImageGenerationTool, ImageGenerationInput


class MockAgent:
    """Mock agent for testing"""

    def __init__(self):
        self.state = {}

    def get_state(self, key):
        return self.state.get(key)

    def set_state(self, key, value):
        self.state[key] = value


@pytest.fixture
def mock_agent():
    """Fixture providing a mock agent"""
    return MockAgent()


@pytest.fixture
def image_tool():
    """Fixture providing an ImageGenerationTool instance"""
    return ImageGenerationTool()


@pytest.fixture
def sample_input():
    """Fixture providing sample input data"""
    return ImageGenerationInput(
        prompt="a beautiful sunset over the ocean",
        width=512,
        height=512,
        num_inference_steps=20,
    )


def test_tool_properties(image_tool):
    """Test basic tool properties"""
    assert image_tool.name == "generate_image"
    assert "image" in image_tool.description.lower()
    assert image_tool.input_schema == ImageGenerationInput


def test_input_validation():
    """Test ImageGenerationInput validation"""
    # Test valid input with just description
    valid_input = ImageGenerationInput(description="test description")
    assert valid_input.description == "test description"

    # Test input with detailed description
    detailed_input = ImageGenerationInput(
        description="A detailed description of a character in a scene with atmospheric lighting"
    )
    assert detailed_input.description == "A detailed description of a character in a scene with atmospheric lighting"


def test_input_validation_constraints():
    """Test input validation constraints"""
    # Test description max length constraint
    with pytest.raises(ValueError):
        ImageGenerationInput(description="x" * 1001)  # Too long (max_length=1000)

    # Test that valid descriptions work
    valid_input = ImageGenerationInput(description="x" * 1000)  # Exactly at limit
    assert len(valid_input.description) == 1000

    # Test that reasonable descriptions work
    normal_input = ImageGenerationInput(description="A normal description")
    assert normal_input.description == "A normal description"

    # Test short descriptions work (empty string is allowed by default in pydantic)
    short_input = ImageGenerationInput(description="")
    assert short_input.description == ""


def test_schema_description(image_tool):
    """Test schema description generation"""
    schema_desc = image_tool.get_schema_description()
    assert "generate_image" in schema_desc
    assert "description" in schema_desc
    assert "required" in schema_desc or "optional" in schema_desc


def test_tool_initialization():
    """Test tool initialization state"""
    tool = ImageGenerationTool()
    assert tool._pipeline is None
    assert tool._model_loaded is False


@pytest.mark.parametrize(
    "description",
    [
        "A portrait of a character",
        "A landscape scene with mountains",
        "A square composition with balanced elements",
        "A detailed character description with specific features and atmospheric lighting",
    ],
)
def test_input_parameter_combinations(description):
    """Test various valid description combinations"""
    input_data = ImageGenerationInput(description=description)
    assert input_data.description == description
