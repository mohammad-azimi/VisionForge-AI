from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
import inspect
import json
import os
import re

import torch
from PIL import Image


PIPELINE_CACHE = {}


@dataclass
class GenerationConfig:
    model_id: str = os.getenv("VISIONFORGE_MODEL_ID", "hf-internal-testing/tiny-stable-diffusion-pipe")
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42
    output_dir: str = "outputs"
    project_name: str = "VisionForge-AI"
    output_type: str = "Portfolio Project Cover"
    style: str = "Premium Minimal"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "visionforge"


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


def build_pipeline_kwargs(
    pipeline,
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
    generator: torch.Generator,
) -> dict:
    accepted_params = set(inspect.signature(pipeline.__call__).parameters.keys())

    candidate_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "generator": generator,
    }

    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in accepted_params
    }


def save_generation(
    image: Image.Image,
    metadata: dict,
    output_dir: str,
    project_name: str,
    seed: int,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    project_slug = slugify(project_name)

    image_path = output_path / f"{project_slug}_{timestamp}_seed-{seed}.png"
    metadata_path = output_path / f"{project_slug}_{timestamp}_seed-{seed}.json"

    image.save(image_path)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return image_path, metadata_path


def generate_image(
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
):
    device = get_device()
    pipeline = load_pipeline(config.model_id)

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.seed)

    pipeline_kwargs = build_pipeline_kwargs(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        config=config,
        generator=generator,
    )

    result = pipeline(**pipeline_kwargs)
    image = result.images[0]

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "device": device,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "favorite": False,
        **asdict(config),
    }

    image_path, metadata_path = save_generation(
        image=image,
        metadata=metadata,
        output_dir=config.output_dir,
        project_name=config.project_name,
        seed=config.seed,
    )

    metadata["image_path"] = str(image_path)
    metadata["metadata_path"] = str(metadata_path)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return image, image_path, metadata


def generate_batch(
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
    batch_size: int,
) -> list[dict]:
    records = []

    for index in range(batch_size):
        current_config = replace(config, seed=config.seed + index)

        image, image_path, metadata = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=current_config,
        )

        records.append(
            {
                "image": image,
                "image_path": image_path,
                "metadata": metadata,
            }
        )

    return records
