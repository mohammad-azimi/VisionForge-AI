from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import os

import torch
from PIL import Image


PIPELINE_CACHE = {}


@dataclass
class GenerationConfig:
    model_id: str = os.getenv("VISIONFORGE_MODEL_ID", "segmind/tiny-sd")
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42
    output_dir: str = "outputs"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_pipeline(model_id: str):
    from diffusers import AutoPipelineForText2Image

    device = get_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = f"{model_id}-{device}-{torch_dtype}"

    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    pipeline = pipeline.to(device)

    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

    PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def save_generation(image: Image.Image, metadata: dict, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_path / f"visionforge_{timestamp}.png"
    metadata_path = output_path / f"visionforge_{timestamp}.json"

    image.save(image_path)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return image_path


def generate_image(
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
):
    device = get_device()
    pipeline = load_pipeline(config.model_id)

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.seed)

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=config.width,
        height=config.height,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )

    image = result.images[0]

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "device": device,
        **asdict(config),
    }

    image_path = save_generation(
        image=image,
        metadata=metadata,
        output_dir=config.output_dir,
    )

    return image, image_path, metadata
