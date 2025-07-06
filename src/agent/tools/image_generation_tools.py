"""
Image generation tools using diffusers and Civitai models
"""

import time
import uuid
from typing import Type, Callable, Any, Optional
from pathlib import Path
from pydantic import Field

from agent.tools import (
    BaseTool,
    ToolInput,
)
from agent.types import (
    ToolResult,
    ToolCallSuccess,
    ToolCallError,
    ImageGenerationToolContent,
)
from agent.paths import agent_paths


class ImageGenerationInput(ToolInput):
    prompt: str = Field(description="Text prompt describing the image to generate")
    negative_prompt: str = Field(
        default="", description="Negative prompt (what to avoid)"
    )
    width: int = Field(
        default=512, description="Image width in pixels", ge=256, le=1024
    )
    height: int = Field(
        default=512, description="Image height in pixels", ge=256, le=1024
    )
    num_inference_steps: int = Field(
        default=20, description="Number of denoising steps", ge=10, le=50
    )
    guidance_scale: float = Field(
        default=7.5, description="How closely to follow the prompt", ge=1.0, le=20.0
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )


class ImageGenerationTool(BaseTool):
    """Tool for generating images using diffusers and Civitai models"""

    def __init__(self):
        self._pipeline = None
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return "Generate an image from a text prompt using AI"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return ImageGenerationInput

    def _load_model(self, progress_callback: Callable[[Any], None]) -> bool:
        """Load the Civitai model using diffusers"""
        if self._model_loaded:
            return True

        try:
            progress_callback({"stage": "loading_dependencies", "progress": 0.1})

            # Import diffusers components - use SDXL for Illustrious models
            from diffusers import StableDiffusionXLPipeline
            import torch

            progress_callback({"stage": "checking_model", "progress": 0.2})

            # Find the Civitai model
            models_dir = agent_paths.get_models_dir()
            model_files = list(models_dir.glob("*.safetensors"))

            if not model_files:
                return False

            model_path = model_files[0]  # Use first .safetensors file found
            progress_callback(
                {"stage": "found_model", "progress": 0.3, "model_path": str(model_path)}
            )

            # Check if we have CUDA available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            progress_callback(
                {"stage": "setting_device", "progress": 0.0, "device": device}
            )

            # Load the pipeline
            progress_callback({"stage": "loading_pipeline", "progress": 0.5})

            # For safetensors files from Civitai, we need to use from_single_file
            # Use SDXL pipeline for Illustrious models
            self._pipeline = StableDiffusionXLPipeline.from_single_file(
                str(model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
            )

            progress_callback({"stage": "moving_to_device", "progress": 0.8})
            self._pipeline = self._pipeline.to(device)

            # Enable memory efficient attention if on CUDA
            if device == "cuda":
                try:
                    self._pipeline.enable_attention_slicing()
                    self._pipeline.enable_sequential_cpu_offload()
                except:
                    pass  # These optimizations are optional

            progress_callback({"stage": "model_ready", "progress": 1.0})
            self._model_loaded = True
            return True

        except ImportError as e:
            progress_callback(
                {
                    "stage": "error",
                    "error": f"Missing dependencies: {str(e)}. Install with: pip install diffusers torch transformers",
                }
            )
            return False
        except Exception as e:
            progress_callback(
                {"stage": "error", "error": f"Failed to load model: {str(e)}"}
            )
            return False

    def _generate_image(
        self, input_data: ImageGenerationInput, progress_callback: Callable[[Any], None]
    ) -> Optional[str]:
        """Generate image and return the file path"""
        try:
            import torch

            # Set random seed if provided
            if input_data.seed is not None:
                torch.manual_seed(input_data.seed)
                progress_callback(
                    {"stage": "seed_set", "progress": 0.1, "seed": input_data.seed}
                )

            progress_callback({"stage": "generating", "progress": 0.2})

            # Generate the image
            with torch.inference_mode():
                result = self._pipeline(
                    prompt=input_data.prompt,
                    negative_prompt=input_data.negative_prompt or None,
                    width=input_data.width,
                    height=input_data.height,
                    num_inference_steps=input_data.num_inference_steps,
                    guidance_scale=input_data.guidance_scale,
                )

            progress_callback({"stage": "image_generated", "progress": 0.8})

            # Save the image
            image = result.images[0]

            # Generate unique filename
            timestamp = int(time.time())
            image_id = str(uuid.uuid4())[:8]
            filename = f"generated_{timestamp}_{image_id}.png"

            # Save to generated_images directory
            generated_images_dir = agent_paths.get_generated_images_dir()
            image_path = generated_images_dir / filename

            image.save(image_path)

            progress_callback(
                {
                    "stage": "image_saved",
                    "progress": 1.0,
                    "image_path": str(image_path),
                    "filename": filename,
                }
            )

            return str(image_path)

        except Exception as e:
            progress_callback(
                {
                    "stage": "generation_error",
                    "error": f"Failed to generate image: {str(e)}",
                }
            )
            return None

    def run(
        self,
        agent,
        input_data: ImageGenerationInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        """Generate an image from the given prompt"""

        # Load model if not already loaded
        if not self._model_loaded:
            progress_callback({"stage": "initializing", "progress": 0.0})

            if not self._load_model(progress_callback):
                models_dir = agent_paths.get_models_dir()
                model_files = list(models_dir.glob("*.safetensors"))

                if not model_files:
                    error_msg = f"No .safetensors model files found in {models_dir}. Please download a Civitai model."
                else:
                    error_msg = "Failed to load image generation model. Check dependencies and model file."

                return ToolCallError(error=error_msg)

        # Generate the image
        progress_callback({"stage": "starting_generation", "progress": 0.0})

        image_path = self._generate_image(input_data, progress_callback)

        if image_path is None:
            return ToolCallError(error="Image generation failed")

        # Prepare success response with structured content
        relative_path = agent_paths.get_relative_to_base(Path(image_path))

        # Create structured ImageGenerationToolContent
        image_content = ImageGenerationToolContent(
            type="image_generated",
            prompt=input_data.prompt,
            image_path=str(relative_path),
            image_url=f"/generated_images/{Path(image_path).name}",
            width=input_data.width,
            height=input_data.height,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            negative_prompt=(
                input_data.negative_prompt if input_data.negative_prompt else None
            ),
            seed=input_data.seed,
        )

        return ToolCallSuccess(type="success", content=image_content)


# Tool instance for registration
IMAGE_GENERATION_TOOLS = [
    ImageGenerationTool(),
]
