"""
Integration test for image generation tool - actually generates images
"""

import pytest


from agent.tools.image_generation_tools import ImageGenerationTool, ImageGenerationInput
from agent.paths import agent_paths
from agent.types import ImageGenerationToolContent, ToolResult


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


@pytest.mark.integration
def test_image_generation_integration(image_tool, mock_agent):
    """Integration test that actually generates an image"""

    # Check if model files exist
    models_dir = agent_paths.get_models_dir()
    model_files = list(models_dir.glob("*.safetensors"))

    if not model_files:
        pytest.skip(f"No .safetensors model files found in {models_dir}")

    # Create test input optimized for SDXL/Illustrious model
    input_data = ImageGenerationInput(
        prompt="a simple red circle on white background",
        width=1024,  # SDXL native resolution
        height=1024,
        num_inference_steps=25,  # Appropriate for SDXL
        guidance_scale=5.0,  # Better for SDXL
    )

    # Track all progress callbacks
    progress_calls = []

    def progress_callback(data):
        progress_calls.append(data)
        print(f"Progress: {data}")

    # Actually generate the image
    result: ToolResult = image_tool.run(
        mock_agent, input_data, "integration_test", progress_callback
    )

    # Verify result
    assert result.type == "success"

    # Verify structured content
    assert isinstance(result.content, ImageGenerationToolContent)
    result_data = result.content
    assert result_data.type == "image_generated"
    assert result_data.prompt == input_data.prompt

    # Verify progress callbacks were made
    assert len(progress_calls) > 0, "Expected progress callbacks during generation"

    # Check that we have model loading and generation stages
    stages = [call.get("stage") for call in progress_calls if "stage" in call]
    assert any(
        "loading" in stage for stage in stages
    ), f"Expected loading stage in: {stages}"
    assert any(
        "generat" in stage for stage in stages
    ), f"Expected generation stage in: {stages}"

    # Verify image was actually saved
    generated_images_dir = agent_paths.get_generated_images_dir()
    png_files = list(generated_images_dir.glob("generated_*.png"))
    assert len(png_files) > 0, f"No generated images found in {generated_images_dir}"

    # Verify the most recent image exists and has reasonable size
    latest_image = max(png_files, key=lambda p: p.stat().st_mtime)
    assert latest_image.exists(), f"Generated image file does not exist: {latest_image}"
    assert (
        latest_image.stat().st_size > 1000
    ), f"Generated image file is too small: {latest_image.stat().st_size} bytes"

    print(f"✅ Successfully generated image: {latest_image}")


@pytest.mark.integration
def test_image_generation_with_seed(image_tool, mock_agent):
    """Test that seeded generation produces consistent results"""

    # Check if model files exist
    models_dir = agent_paths.get_models_dir()
    model_files = list(models_dir.glob("*.safetensors"))

    if not model_files:
        pytest.skip(f"No .safetensors model files found in {models_dir}")

    # Use same seed twice
    seed = 42
    input_data = ImageGenerationInput(
        prompt="a blue square", width=256, height=256, num_inference_steps=10, seed=seed
    )

    progress_calls = []

    def progress_callback(data):
        progress_calls.append(data)

    # Generate first image
    result1: ToolResult = image_tool.run(
        mock_agent, input_data, "seed_test_1", progress_callback
    )

    # Generate second image with same seed
    progress_calls.clear()
    result2: ToolResult = image_tool.run(
        mock_agent, input_data, "seed_test_2", progress_callback
    )

    # Both should succeed and have structured content
    assert result1.type == "success"
    assert result2.type == "success"

    assert result1.content.type == "image_generated"
    assert result2.content.type == "image_generated"
    assert result1.content.seed == seed
    assert result2.content.seed == seed

    print("✅ Seeded generation test completed")


@pytest.mark.integration
def test_image_generation_with_negative_prompt(image_tool, mock_agent):
    """Test image generation with negative prompt"""

    # Check if model files exist
    models_dir = agent_paths.get_models_dir()
    model_files = list(models_dir.glob("*.safetensors"))

    if not model_files:
        pytest.skip(f"No .safetensors model files found in {models_dir}")

    input_data = ImageGenerationInput(
        prompt="a beautiful landscape",
        negative_prompt="ugly, blurry, low quality",
        width=256,
        height=256,
        num_inference_steps=10,
    )

    progress_calls = []

    def progress_callback(data):
        progress_calls.append(data)

    result: ToolResult = image_tool.run(
        mock_agent, input_data, "negative_test", progress_callback
    )

    assert result.type == "success"
    assert result.content.type == "image_generated"
    assert result.content.negative_prompt == "ugly, blurry, low quality"

    print("✅ Negative prompt test completed")


@pytest.mark.integration
def test_different_image_sizes(image_tool, mock_agent):
    """Test generation with different image sizes"""

    # Check if model files exist
    models_dir = agent_paths.get_models_dir()
    model_files = list(models_dir.glob("*.safetensors"))

    if not model_files:
        pytest.skip(f"No .safetensors model files found in {models_dir}")

    # Test different sizes
    sizes = [(256, 256), (512, 256), (256, 512)]

    for width, height in sizes:
        input_data = ImageGenerationInput(
            prompt=f"test image {width}x{height}",
            width=width,
            height=height,
            num_inference_steps=10,
        )

        progress_calls = []

        def progress_callback(data):
            progress_calls.append(data)

        result: ToolResult = image_tool.run(
            mock_agent, input_data, f"size_test_{width}x{height}", progress_callback
        )

        assert result.type == "success"
        assert result.content.type == "image_generated"
        assert result.content.width == width
        assert result.content.height == height

    print("✅ Different sizes test completed")
