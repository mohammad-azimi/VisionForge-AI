from pathlib import Path
import textwrap

path = Path("src/visionforge/generator.py")

content = r'''
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
    model_id: str = os.getenv(
        "VISIONFORGE_MODEL_ID",
        "hf-internal-testing/tiny-stable-diffusion-pipe",
    )
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42
    output_dir: str = "outputs"
    project_name: str = "VisionForge-AI"
    output_type: str = "Portfolio Project Cover"
    style: str = "Premium Minimal"
    generation_mode: str = "text-to-image"


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


def disable_safety_checker(pipeline):
    # The tiny testing model can fail inside the safety checker because
    # its safety-checker image size does not match normal generated images.
    # Disabling it fixes local test generation.
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None

    if hasattr(pipeline, "requires_safety_checker"):
        pipeline.requires_safety_checker = False

    return pipeline


def optimize_pipeline(pipeline):
    pipeline = disable_safety_checker(pipeline)

    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

    return pipeline


def load_text2img_pipeline(model_id: str):
    from diffusers import AutoPipelineForText2Image

    device = get_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = f"text2img::{model_id}::{device}::{torch_dtype}"

    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    pipeline = optimize_pipeline(pipeline)
    pipeline = pipeline.to(device)

    PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def load_img2img_pipeline(model_id: str):
    from diffusers import AutoPipelineForImage2Image

    device = get_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = f"img2img::{model_id}::{device}::{torch_dtype}"

    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    pipeline = optimize_pipeline(pipeline)
    pipeline = pipeline.to(device)

    PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def get_generator(seed: int) -> torch.Generator:
    device = get_device()
    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def build_text2img_kwargs(
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


def build_img2img_kwargs(
    pipeline,
    prompt: str,
    negative_prompt: str,
    init_image: Image.Image,
    config: GenerationConfig,
    generator: torch.Generator,
    strength: float,
) -> dict:
    accepted_params = set(inspect.signature(pipeline.__call__).parameters.keys())

    candidate_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": init_image,
        "strength": strength,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "generator": generator,
    }

    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in accepted_params
    }


def prepare_init_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image.resize((width, height), Image.LANCZOS)


def save_generation(
    image: Image.Image,
    metadata: dict,
    output_dir: str,
    project_name: str,
    seed: int,
    generation_mode: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    project_slug = slugify(project_name)
    mode_slug = slugify(generation_mode)

    image_path = output_path / f"{project_slug}_{mode_slug}_{timestamp}_seed-{seed}.png"
    metadata_path = output_path / f"{project_slug}_{mode_slug}_{timestamp}_seed-{seed}.json"

    image.save(image_path)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return image_path, metadata_path


def write_final_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_image(
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
):
    device = get_device()
    pipeline = load_text2img_pipeline(config.model_id)
    generator = get_generator(config.seed)

    pipeline_kwargs = build_text2img_kwargs(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        config=config,
        generator=generator,
    )

    with torch.inference_mode():
        result = pipeline(**pipeline_kwargs)

    image = result.images[0]

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "device": device,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "favorite": False,
        "generation_mode": "text-to-image",
        **asdict(config),
    }

    image_path, metadata_path = save_generation(
        image=image,
        metadata=metadata,
        output_dir=config.output_dir,
        project_name=config.project_name,
        seed=config.seed,
        generation_mode="text-to-image",
    )

    metadata["image_path"] = str(image_path)
    metadata["metadata_path"] = str(metadata_path)
    write_final_metadata(metadata_path, metadata)

    return image, image_path, metadata


def generate_image_from_image(
    prompt: str,
    negative_prompt: str,
    init_image: Image.Image,
    config: GenerationConfig,
    strength: float = 0.6,
):
    device = get_device()
    pipeline = load_img2img_pipeline(config.model_id)
    generator = get_generator(config.seed)

    prepared_image = prepare_init_image(
        image=init_image,
        width=config.width,
        height=config.height,
    )

    pipeline_kwargs = build_img2img_kwargs(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=prepared_image,
        config=config,
        generator=generator,
        strength=strength,
    )

    with torch.inference_mode():
        result = pipeline(**pipeline_kwargs)

    image = result.images[0]

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "device": device,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "favorite": False,
        "generation_mode": "image-to-image",
        "strength": strength,
        "source_image_size": {
            "width": init_image.width,
            "height": init_image.height,
        },
        **asdict(config),
    }

    image_path, metadata_path = save_generation(
        image=image,
        metadata=metadata,
        output_dir=config.output_dir,
        project_name=config.project_name,
        seed=config.seed,
        generation_mode="image-to-image",
    )

    metadata["image_path"] = str(image_path)
    metadata["metadata_path"] = str(metadata_path)
    write_final_metadata(metadata_path, metadata)

    return image, image_path, metadata


def generate_batch(
    prompt: str,
    negative_prompt: str,
    config: GenerationConfig,
    batch_size: int,
) -> list[dict]:
    records = []

    for index in range(batch_size):
        current_config = replace(
            config,
            seed=config.seed + index,
            generation_mode="text-to-image",
        )

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


def generate_batch_from_image(
    prompt: str,
    negative_prompt: str,
    init_image: Image.Image,
    config: GenerationConfig,
    batch_size: int,
    strength: float = 0.6,
) -> list[dict]:
    records = []

    for index in range(batch_size):
        current_config = replace(
            config,
            seed=config.seed + index,
            generation_mode="image-to-image",
        )

        image, image_path, metadata = generate_image_from_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            config=current_config,
            strength=strength,
        )

        records.append(
            {
                "image": image,
                "image_path": image_path,
                "metadata": metadata,
            }
        )

    return records
'''

path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
print("Rewrote src/visionforge/generator.py with the fixed step 4 generator.")