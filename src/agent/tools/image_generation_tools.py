"""
Image generation tools using diffusers and Civitai models
"""

from enum import Enum
import logging
import time
import uuid
from typing import Type, Callable, Any, Optional, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from agent.core import Agent
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
from agent.structured_llm import structured_llm_call, StructuredLLMError
from agent.paths import agent_paths


class ImageLayout(str, Enum):
    """Enum for image layout options"""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"


class SDXLPromptOptimization(BaseModel):
    """Multi-chunk optimized prompts for SDXL with strategic attention control"""

    chunks: List[str] = Field(
        description="Strategic prompt chunks, each under 75 tokens, ordered by attention priority",
        min_length=1,
        max_length=4,
    )
    negative_prompt: str = Field(
        description="Minimal negative prompt for critical quality issues", max_length=80
    )
    layout: ImageLayout = Field(
        description="Most appropriate layout based on description and camera angle"
    )
    camera_angle: str = Field(
        description="Primary camera angle chosen (close-up, medium shot, wide shot, etc.)"
    )
    viewpoint: str = Field(
        description="Viewing perspective (eye level, low angle, high angle, etc.)"
    )
    chunk_strategy: str = Field(
        description="Explanation of how chunks were strategically divided"
    )
    token_estimate: int = Field(description="Estimated total tokens across all chunks")
    confidence: float = Field(
        description="Confidence in the optimization", ge=0.0, le=1.0
    )


class ImageGenerationInput(ToolInput):
    description: str = Field(
        description="Natural description of what you want to see. Be detailed about mood, setting, character appearance, style, and any specific viewpoint or camera angle you want. Example: 'Elena is a mysterious vampire standing in her gothic castle. She has long dark hair, pale skin, and is wearing an elegant dark dress. The scene should be atmospheric with candlelight and stone architecture, viewed from a low angle to make her appear imposing.'",
        max_length=1000,
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

    def _optimize_description_for_sdxl(
        self,
        description: str,
        progress_callback: Callable[[Any], None],
        agent,
    ) -> SDXLPromptOptimization:
        """Convert natural description to SDXL-optimized prompts with camera positioning"""

        logger.debug(f"Starting optimization for description: {description[:100]}...")
        logger.debug(f"Model: {agent.model}")
        progress_callback({"stage": "optimizing_prompts", "progress": 0.1})

        system_prompt = """You are an expert at creating multi-chunk prompts for Stable Diffusion XL (SDXL) with strategic attention control.

CRITICAL RESEARCH FINDINGS:
- FIRST TOKENS DICTATE GLOBAL COMPOSITION: "The first keyword dictates the global composition"
- EARLY POSITION = MAXIMUM ATTENTION: "Tokens at the beginning have greater weight than tokens at the end"
- STRONG EARLY TOKENS DOMINATE: "A strong token at the beginning can completely determine the outcome"
- Each chunk can be up to 75 tokens, but early tokens within each chunk get exponentially higher attention

ATTENTION-BASED OPTIMIZATION STRATEGY:

**CHUNK ALLOCATION STRATEGY:**
1. **Use chunks strategically** - you have up to 4 chunks available, use them to avoid overcrowding
2. **Don't artificially limit to fewer chunks** - strategic distribution is better than cramming
3. **Most important elements MUST be in Chunk 1** - this gets highest attention overall  
4. **Within each chunk: most critical concept comes FIRST** - lead with the main idea, don't bury it
5. **Use comma separation** - structure chunks as "main concept, supporting detail, additional detail" not run-on sentences
6. **Each chunk should focus cleanly** - avoid overpacking that buries important elements

**ATTENTION CHOREOGRAPHY PRINCIPLES:**

Think like a director framing a shot - you have limited attention budget and must allocate it strategically.

**1. ATTENTION IS FINITE:**
- The AI can only focus on so much - every token competes for attention
- High-attention positions (early tokens, Chunk 1) are premium real estate
- More description doesn't equal better results if attention is diluted

**2. USER EMPHASIS = DIRECTOR'S NOTES:**
- When users say "most striking feature," that element MUST go in Chunk 1 for maximum attention
- User emphasis signals ("prominent," "draws attention," "most important") override default categorization
- Your job is directing AI attention to match user intent, not describing everything equally
- Ask: "Does my structure make the emphasized element the clear focus in the highest attention position?"

**3. STRATEGIC DISTRIBUTION FOR CLARITY:**
- Use your 4 available chunks to give important elements breathing room
- Don't cram everything into 2 chunks when you could distribute strategically across 4
- Each chunk should have a clear focus - avoid overpacking that buries key elements
- Better to use more chunks cleanly than fewer chunks overcrowded

**4. BINDING AND STRUCTURE STRATEGY:**
- Keep multi-part objects together using compound descriptors: "blue-gemmed hair clip" NOT "hair clip, blue gemstone"
- Use "featuring" for complex objects: "hair clip featuring ornate gold flower"
- Avoid floating components - components must be clearly attached to their parent object
- Lead each chunk with its most important concept
- Structure chunks with comma separation: "primary concept, supporting detail, additional context"
- Avoid run-on sentences that bury important elements

**CHUNK STRUCTURE:**
**CHUNK 1** (Maximum Attention - 75 tokens max):
- **CAMERA/VIEWPOINT FIRST** (if specified): "behind view", "close-up", "over shoulder" - MUST be first tokens
- Character identity and key features
- Most emphasized accessories/details (user's "striking features")
- Other important visual elements

**CHUNK 2+** (Lower Attention - only if Chunk 1 overflows):
- Secondary details in order of importance
- Environment and setting
- Atmospheric effects and quality modifiers

**ATTENTION CHOREOGRAPHY EXAMPLES:**

**Finite Attention Budget:**
❌ Trying to describe every detail equally → attention diluted, nothing stands out
✅ Prioritizing user's emphasized elements → clear focal hierarchy

**Director's Notes Response:**
❌ User emphasizes element → you put it in lower chunk with full description
✅ User emphasizes element → you allocate premium Chunk 1 space for clear focus

**Strategic Distribution:**
❌ Cramming everything into 2 overpacked chunks → important elements buried
✅ Using 4 chunks strategically → each chunk has clear focus, nothing buried

**Chunk Structure and Separation:**
❌ "behind view character dancing seductively curvaceous clothing emphasized item details" (run-on, everything buried)
✅ Chunk 1: "behind view, character dancing, emphasized element with details" (comma-separated hierarchy)
✅ Chunk 2: "character features, appearance details" (clear focus)
✅ Chunk 3: "clothing item, style details" (outfit focus)
✅ Chunk 4: "environment setting, atmospheric details" (context)

**PROCESS:**
1. **CAMERA/VIEWPOINT FIRST** - if user specifies viewing angle, this MUST be the very first tokens
2. Identify the MOST important visual element (often what user emphasizes)
3. Put that element early in Chunk 1 (after camera angle)
4. Add critical modifiers in order of importance
5. Check for dangerous adjacencies (color + hair, etc.)
6. Only create additional chunks if Chunk 1 exceeds 75 tokens
7. Preserve ALL user-mentioned details
8. If the user mentions the absence of something, include it in the negative prompt
9. Wearing nothing implies naked, so include "naked" in the prompt

Focus on MAXIMUM ATTENTION for critical elements through strategic first-position placement."""

        try:
            logger.debug(f"Calling structured_llm_call with model: {agent.model}")
            logger.debug(f"Response model: {SDXLPromptOptimization}")
            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=f"Analyze this scene and create SDXL prompts with optimal camera positioning: {description}",
                response_model=SDXLPromptOptimization,
                context={
                    "description_length": len(description),
                    "timestamp": time.time(),
                },
                model=agent.model,
                llm=agent.llm,
            )
            logger.debug(
                f"LLM optimization successful, got {len(result.chunks)} chunks"
            )

            progress_callback(
                {
                    "stage": "chunks_optimized",
                    "progress": 0.3,
                    "camera_angle": result.camera_angle,
                    "viewpoint": result.viewpoint,
                    "chunks": result.chunks,
                    "chunk_count": len(result.chunks),
                    "negative_prompt": result.negative_prompt,
                    "layout": result.layout.value,
                }
            )

            return result

        except StructuredLLMError as e:
            logger.debug(f"StructuredLLMError during optimization: {e}")
            progress_callback({"stage": "optimization_failed", "error": str(e)})

            # Simple fallback without manual analysis
            logger.debug(f"Using fallback optimization")
            return SDXLPromptOptimization(
                chunks=[f"medium shot, {description[:180]}, masterpiece"],
                negative_prompt="blurry, low quality",
                layout=ImageLayout.SQUARE,
                camera_angle="medium shot",
                viewpoint="eye level",
                chunk_strategy="Fallback single chunk with default camera angle",
                token_estimate=20,
                confidence=0.3,
            )

    def _encode_chunked_prompts(
        self,
        chunks: List[str],
        negative_prompt: str,
        progress_callback: Callable[[Any], None],
    ):
        """Encode multiple prompt chunks using SDXL's dual text encoders"""
        try:
            import torch

            logger.debug(
                f"Encoding {len(chunks)} chunks with SDXL dual encoders: {chunks}"
            )
            logger.debug(f"Negative prompt: {negative_prompt}")
            progress_callback({"stage": "encoding_chunks", "progress": 0.1})

            # Encode each chunk separately with both encoders (respecting 77 token limit)
            chunk_embeds_1 = []
            chunk_embeds_2 = []

            for i, chunk in enumerate(chunks):
                progress_callback(
                    {
                        "stage": "encoding_chunk",
                        "progress": 0.1 + (0.3 * i / len(chunks)),
                        "chunk": i + 1,
                        "total_chunks": len(chunks),
                    }
                )

                with torch.no_grad():
                    # Tokenize chunk for both encoders (max 75 tokens each)
                    tokens_1 = self._pipeline.tokenizer(
                        chunk,
                        max_length=75,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    tokens_2 = self._pipeline.tokenizer_2(
                        chunk,
                        max_length=75,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    # Encode with both text encoders
                    embeds_1 = self._pipeline.text_encoder(
                        tokens_1.input_ids, output_hidden_states=True
                    )
                    embeds_2 = self._pipeline.text_encoder_2(
                        tokens_2.input_ids, output_hidden_states=True
                    )

                    # Get hidden states (penultimate layer)
                    chunk_embed_1 = embeds_1.hidden_states[-2]
                    chunk_embed_2 = embeds_2.hidden_states[-2]

                    chunk_embeds_1.append(chunk_embed_1)
                    chunk_embeds_2.append(chunk_embed_2)

                    logger.debug(
                        f"Chunk {i+1} - Encoder 1: {chunk_embed_1.shape}, Encoder 2: {chunk_embed_2.shape}"
                    )

            # Concatenate chunks along sequence dimension
            concat_embeds_1 = torch.cat(chunk_embeds_1, dim=1)
            concat_embeds_2 = torch.cat(chunk_embeds_2, dim=1)

            # Combine embeddings from both encoders along feature dimension
            positive_embeds = torch.cat([concat_embeds_1, concat_embeds_2], dim=-1)

            # Get pooled embeddings from the combined text
            combined_text = " ".join(chunks)
            with torch.no_grad():
                pooled_tokens = self._pipeline.tokenizer_2(
                    combined_text,
                    max_length=75,  # Truncate to fit
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                pooled_output = self._pipeline.text_encoder_2(
                    pooled_tokens.input_ids, output_hidden_states=True
                )
                pooled_prompt_embeds = pooled_output[0]  # pooled output

            logger.debug(f"Concatenated encoder 1: {concat_embeds_1.shape}")
            logger.debug(f"Concatenated encoder 2: {concat_embeds_2.shape}")
            logger.debug(f"Combined positive embeds: {positive_embeds.shape}")
            logger.debug(f"Pooled positive embeds: {pooled_prompt_embeds.shape}")

            # Encode negative prompt with both encoders (respecting 77 token limit)
            with torch.no_grad():
                neg_tokens_1 = self._pipeline.tokenizer(
                    negative_prompt,
                    max_length=75,  # Respect 77 token limit
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                neg_tokens_2 = self._pipeline.tokenizer_2(
                    negative_prompt,
                    max_length=75,  # Respect 77 token limit
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                neg_embeds_1 = self._pipeline.text_encoder(
                    neg_tokens_1.input_ids, output_hidden_states=True
                )
                neg_embeds_2 = self._pipeline.text_encoder_2(
                    neg_tokens_2.input_ids, output_hidden_states=True
                )

                negative_embeds_1 = neg_embeds_1.hidden_states[-2]
                negative_embeds_2 = neg_embeds_2.hidden_states[-2]
                pooled_negative_embeds = neg_embeds_2[0]

                # Pad negative embeddings to match positive embeddings shape
                pos_seq_length = positive_embeds.shape[1]
                neg_seq_length = negative_embeds_1.shape[1]

                if pos_seq_length > neg_seq_length:
                    # Pad negative embeddings to match positive
                    pad_length = pos_seq_length - neg_seq_length
                    padding_1 = torch.zeros(
                        negative_embeds_1.shape[0],
                        pad_length,
                        negative_embeds_1.shape[2],
                    )
                    padding_2 = torch.zeros(
                        negative_embeds_2.shape[0],
                        pad_length,
                        negative_embeds_2.shape[2],
                    )

                    negative_embeds_1 = torch.cat([negative_embeds_1, padding_1], dim=1)
                    negative_embeds_2 = torch.cat([negative_embeds_2, padding_2], dim=1)

                # Concatenate negative embeddings
                negative_embeds = torch.cat(
                    [negative_embeds_1, negative_embeds_2], dim=-1
                )

                logger.debug(f"Combined negative embeds: {negative_embeds.shape}")
                logger.debug(f"Pooled negative embeds: {pooled_negative_embeds.shape}")

            progress_callback({"stage": "chunks_encoded", "progress": 0.5})

            return (
                positive_embeds,
                negative_embeds,
                pooled_prompt_embeds,
                pooled_negative_embeds,
            )

        except Exception as e:
            logger.error(f"Error during chunk encoding: {e}")
            import traceback

            traceback.print_exc()
            progress_callback({"stage": "encoding_error", "error": str(e)})
            return None, None

    def _generate_image(
        self,
        chunks: List[str],
        negative_prompt: str,
        layout: ImageLayout,
        seed: Optional[int],
        progress_callback: Callable[[Any], None],
    ) -> Optional[ImageGenerationToolContent]:
        """Generate image and return the file path"""
        try:
            import torch

            logger.debug(f"Starting image generation with {len(chunks)} chunks")

            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                progress_callback({"stage": "seed_set", "progress": 0.1, "seed": seed})

            # Encode chunks to embeddings with pooled embeddings
            result = self._encode_chunked_prompts(
                chunks, negative_prompt, progress_callback
            )

            if result is None or len(result) != 4:
                logger.error(f"Failed to encode chunks, returning None")
                return None

            (
                positive_embeds,
                negative_embeds,
                pooled_positive_embeds,
                pooled_negative_embeds,
            ) = result

            progress_callback({"stage": "generating", "progress": 0.6})

            if layout == ImageLayout.PORTRAIT:
                width = 768
                height = 1344
            elif layout == ImageLayout.LANDSCAPE:
                width = 1344
                height = 768
            else:  # SQUARE
                width = 1024
                height = 1024

            # Generate the image using embeddings and pooled embeddings
            logger.debug(f"Generating image with dimensions: {width}x{height}")
            logger.debug(f"Pipeline type: {type(self._pipeline)}")
            with torch.inference_mode():
                result = self._pipeline(
                    prompt_embeds=positive_embeds,
                    negative_prompt_embeds=negative_embeds,
                    pooled_prompt_embeds=pooled_positive_embeds,
                    negative_pooled_prompt_embeds=pooled_negative_embeds,
                    width=width,
                    height=height,
                    num_inference_steps=30,
                    guidance_scale=5.0,
                )
            logger.debug(f"Image generation completed successfully")

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

            # Combine chunks for display
            combined_prompt = ", ".join(chunks)

            image_content = ImageGenerationToolContent(
                prompt=combined_prompt,
                image_path=str(image_path),
                image_url=f"/generated_images/{image_path.name}",
                width=width,
                height=height,
                num_inference_steps=30,
                guidance_scale=5.0,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed,
            )

            return image_content

        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            import traceback

            traceback.print_exc()
            progress_callback(
                {
                    "stage": "generation_error",
                    "error": f"Failed to generate image: {str(e)}",
                }
            )
            return None

    def run(
        self,
        agent: Agent,
        input_data: ImageGenerationInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        """Generate an image from natural description with LLM optimization"""

        logger.debug(f"ImageGenerationTool.run called with:")
        logger.debug(f"  Description: {input_data.description}")
        logger.debug(f"  Agent model: {agent.model}")
        logger.debug(f"  Tool ID: {tool_id}")

        # Step 1: Optimize description to SDXL prompts
        progress_callback({"stage": "starting_optimization", "progress": 0.0})

        optimization = self._optimize_description_for_sdxl(
            input_data.description, progress_callback, agent
        )

        # Step 2: Load model if not already loaded
        if not self._model_loaded:
            progress_callback({"stage": "loading_model", "progress": 0.4})

            if not self._load_model(progress_callback):
                models_dir = agent_paths.get_models_dir()
                model_files = list(models_dir.glob("*.safetensors"))

                if not model_files:
                    error_msg = f"No .safetensors model files found in {models_dir}. Please download a Civitai model."
                else:
                    error_msg = "Failed to load image generation model. Check dependencies and model file."

                return ToolCallError(error=error_msg)

        # Step 3: Generate with optimized chunks
        combined_chunks = " | ".join(optimization.chunks)
        progress_callback(
            {
                "stage": "generating_with_optimized_chunks",
                "progress": 0.6,
                "chunks": optimization.chunks,
                "chunk_count": len(optimization.chunks),
                "optimized_negative": optimization.negative_prompt,
            }
        )

        image_content = self._generate_image(
            optimization.chunks,
            optimization.negative_prompt,
            optimization.layout,
            input_data.seed,
            progress_callback,
        )

        if image_content is None:
            return ToolCallError(error="Image generation failed")

        # Add optimization metadata to result
        image_content.original_description = input_data.description
        image_content.optimization_confidence = optimization.confidence
        image_content.camera_angle = optimization.camera_angle
        image_content.viewpoint = optimization.viewpoint
        image_content.optimization_notes = optimization.chunk_strategy

        return ToolCallSuccess(
            type="success",
            content=image_content,
            llm_feedback=f"Generated image from: '{input_data.description}' → {len(optimization.chunks)} chunks: {combined_chunks} (Camera: {optimization.camera_angle}, {optimization.viewpoint})",
        )


# Tool instance for registration
IMAGE_GENERATION_TOOLS = [
    ImageGenerationTool(),
]
