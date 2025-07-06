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
    # Test valid input
    valid_input = ImageGenerationInput(prompt="test prompt", width=512, height=512)
    assert valid_input.prompt == "test prompt"
    assert valid_input.width == 512
    assert valid_input.height == 512
    assert valid_input.num_inference_steps == 20  # default

    # Test input with all fields
    full_input = ImageGenerationInput(
        prompt="detailed prompt",
        negative_prompt="bad quality",
        width=768,
        height=768,
        num_inference_steps=30,
        guidance_scale=8.0,
        seed=42,
    )
    assert full_input.negative_prompt == "bad quality"
    assert full_input.seed == 42


def test_input_validation_constraints():
    """Test input validation constraints"""
    # Test width/height constraints
    with pytest.raises(ValueError):
        ImageGenerationInput(prompt="test", width=100)  # Too small

    with pytest.raises(ValueError):
        ImageGenerationInput(prompt="test", width=2000)  # Too large

    # Test steps constraints
    with pytest.raises(ValueError):
        ImageGenerationInput(prompt="test", num_inference_steps=5)  # Too few

    with pytest.raises(ValueError):
        ImageGenerationInput(prompt="test", num_inference_steps=100)  # Too many


def test_schema_description(image_tool):
    """Test schema description generation"""
    schema_desc = image_tool.get_schema_description()
    assert "generate_image" in schema_desc
    assert "prompt" in schema_desc
    assert "required" in schema_desc or "optional" in schema_desc


def test_tool_initialization():
    """Test tool initialization state"""
    tool = ImageGenerationTool()
    assert tool._pipeline is None
    assert tool._model_loaded is False


@pytest.mark.parametrize(
    "width,height,steps",
    [
        (256, 256, 10),
        (512, 512, 20),
        (768, 768, 30),
        (1024, 1024, 50),
    ],
)
def test_input_parameter_combinations(width, height, steps):
    """Test various valid parameter combinations"""
    input_data = ImageGenerationInput(
        prompt="test image", width=width, height=height, num_inference_steps=steps
    )
    assert input_data.width == width
    assert input_data.height == height
    assert input_data.num_inference_steps == steps
