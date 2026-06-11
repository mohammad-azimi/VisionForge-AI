from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resize_for_eval(image: Image.Image, size: int = 256) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def clamp_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def compute_brightness_score(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    mean_brightness = float(gray.mean())
    score = 100.0 - abs(mean_brightness - 0.50) * 220.0
    return clamp_score(score)


def compute_contrast_score(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    contrast = float(gray.std())

    if contrast < 0.03:
        return clamp_score(contrast / 0.03 * 30.0)

    if contrast <= 0.23:
        return clamp_score(45.0 + contrast / 0.23 * 55.0)

    return clamp_score(100.0 - (contrast - 0.23) * 160.0)


def compute_color_score(arr: np.ndarray) -> float:
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = float((max_channel - min_channel).mean())

    if saturation < 0.04:
        return clamp_score(saturation / 0.04 * 45.0)

    if saturation <= 0.32:
        return clamp_score(55.0 + saturation / 0.32 * 45.0)

    return clamp_score(100.0 - (saturation - 0.32) * 120.0)


def compute_sharpness_score(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)

    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    gradient = float((grad_x + grad_y) / 2.0)

    if gradient < 0.015:
        return clamp_score(gradient / 0.015 * 45.0)

    if gradient <= 0.080:
        return clamp_score(45.0 + gradient / 0.080 * 55.0)

    return clamp_score(100.0 - (gradient - 0.080) * 500.0)


def compute_center_focus_score(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    height, width = gray.shape

    y, x = np.ogrid[:height, :width]
    center_y = height / 2.0
    center_x = width / 2.0

    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    center_mask = distance <= min(width, height) * 0.30
    border_mask = distance >= min(width, height) * 0.43

    center_energy = float(gray[center_mask].std() + gray[center_mask].mean())
    border_energy = float(gray[border_mask].std() + gray[border_mask].mean())

    if border_energy <= 0.001:
        return 85.0

    ratio = center_energy / border_energy
    return clamp_score(45.0 + ratio * 30.0)


def compute_clutter_penalty(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))

    edge_density_x = float((grad_x > 0.10).mean())
    edge_density_y = float((grad_y > 0.10).mean())
    edge_density = (edge_density_x + edge_density_y) / 2.0

    if edge_density <= 0.08:
        return 0.0

    return clamp_score((edge_density - 0.08) * 350.0)


def compute_radial_artifact_penalty(arr: np.ndarray) -> float:
    """
    Penalizes repeated circular / eye-like / portal-like patterns.

    This is optional because some projects intentionally use circular symbols.
    """
    gray = arr.mean(axis=2)
    height, width = gray.shape
    center_y = height / 2.0
    center_x = width / 2.0

    y, x = np.indices(gray.shape)
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    radius = radius.astype(np.int32)

    max_radius = int(min(width, height) * 0.48)
    radial_profile = []

    for r in range(2, max_radius):
        mask = radius == r
        if mask.any():
            radial_profile.append(float(gray[mask].mean()))

    if len(radial_profile) < 16:
        return 0.0

    profile = np.asarray(radial_profile, dtype=np.float32)
    profile = profile - profile.mean()

    radial_edges = np.abs(np.diff(profile))
    strong_ring_ratio = float((radial_edges > 0.030).mean())
    profile_variation = float(profile.std())

    penalty = 0.0

    if strong_ring_ratio > 0.12:
        penalty += (strong_ring_ratio - 0.12) * 260.0

    if profile_variation > 0.12:
        penalty += (profile_variation - 0.12) * 220.0

    return clamp_score(penalty)


def compute_pixel_histogram_reference_similarity(
    image_path: str | Path,
    reference_image_path: str | Path | None,
) -> float | None:
    if not reference_image_path:
        return None

    image = resize_for_eval(load_rgb_image(image_path), size=224)
    reference = resize_for_eval(load_rgb_image(reference_image_path), size=224)

    arr = image_to_array(image)
    ref = image_to_array(reference)

    mean_absolute_error = float(np.abs(arr - ref).mean())
    pixel_similarity = 100.0 * (1.0 - mean_absolute_error)

    hist_scores = []

    for channel in range(3):
        hist_a, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(ref[:, :, channel], bins=32, range=(0.0, 1.0), density=True)

        hist_a = hist_a / (hist_a.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)

        overlap = np.minimum(hist_a, hist_b).sum()
        hist_scores.append(float(overlap * 100.0))

    histogram_similarity = float(np.mean(hist_scores))
    return clamp_score(pixel_similarity * 0.55 + histogram_similarity * 0.45)


def get_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache(maxsize=2)
def load_clip_model(model_id: str = DEFAULT_CLIP_MODEL_ID):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = get_torch_device()
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    return model, processor, device, torch


def normalized_clip_features_for_image(image: Image.Image, model_id: str = DEFAULT_CLIP_MODEL_ID):
    model, processor, device, torch = load_clip_model(model_id)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features


def normalized_clip_features_for_text(text: str, model_id: str = DEFAULT_CLIP_MODEL_ID):
    model, processor, device, torch = load_clip_model(model_id)
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features


def clip_cosine_to_score(cosine_value: float, mode: str) -> float:
    """
    Converts CLIP cosine similarity into a practical ranking score.
    These scores are calibrated for ranking candidates, not absolute scientific measurement.
    """
    if mode == "text":
        return clamp_score((cosine_value - 0.15) / 0.20 * 100.0)

    if mode == "image":
        return clamp_score((cosine_value - 0.55) / 0.35 * 100.0)

    return clamp_score((cosine_value + 1.0) * 50.0)


def compute_clip_prompt_alignment(
    image_path: str | Path,
    prompt: str | None,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
) -> float | None:
    if not prompt:
        return None

    image = resize_for_eval(load_rgb_image(image_path), size=224)

    image_features = normalized_clip_features_for_image(image, model_id=clip_model_id)
    text_features = normalized_clip_features_for_text(prompt, model_id=clip_model_id)

    cosine = float((image_features @ text_features.T).item())
    return clip_cosine_to_score(cosine, mode="text")


def compute_clip_reference_similarity(
    image_path: str | Path,
    reference_image_path: str | Path | None,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
) -> float | None:
    if not reference_image_path:
        return None

    image = resize_for_eval(load_rgb_image(image_path), size=224)
    reference = resize_for_eval(load_rgb_image(reference_image_path), size=224)

    image_features = normalized_clip_features_for_image(image, model_id=clip_model_id)
    reference_features = normalized_clip_features_for_image(reference, model_id=clip_model_id)

    cosine = float((image_features @ reference_features.T).item())
    return clip_cosine_to_score(cosine, mode="image")


def combine_scores(
    visual_quality_score: float,
    prompt_alignment_score: float | None,
    reference_similarity: float | None,
    clip_reference_similarity: float | None,
    evaluation_profile: str,
) -> float:
    profile = evaluation_profile.lower().strip()

    prompt_score = prompt_alignment_score
    ref_score = reference_similarity

    if clip_reference_similarity is not None and reference_similarity is not None:
        ref_score = reference_similarity * 0.35 + clip_reference_similarity * 0.65
    elif clip_reference_similarity is not None:
        ref_score = clip_reference_similarity

    if profile == "reference_match":
        score = visual_quality_score * 0.25

        if ref_score is not None:
            score += ref_score * 0.60
        else:
            score += visual_quality_score * 0.35

        if prompt_score is not None:
            score += prompt_score * 0.15
        else:
            score += visual_quality_score * 0.15

        return clamp_score(score)

    if profile == "portrait":
        score = visual_quality_score * 0.40

        if prompt_score is not None:
            score += prompt_score * 0.35
        else:
            score += visual_quality_score * 0.20

        if ref_score is not None:
            score += ref_score * 0.25
        else:
            score += visual_quality_score * 0.25

        return clamp_score(score)

    if profile == "general":
        score = visual_quality_score * 0.55

        if prompt_score is not None:
            score += prompt_score * 0.35
        else:
            score += visual_quality_score * 0.25

        if ref_score is not None:
            score += ref_score * 0.10
        else:
            score += visual_quality_score * 0.10

        return clamp_score(score)

    # Default: portfolio cover
    score = visual_quality_score * 0.65

    if prompt_score is not None:
        score += prompt_score * 0.25
    else:
        score += visual_quality_score * 0.20

    if ref_score is not None:
        score += ref_score * 0.10
    else:
        score += visual_quality_score * 0.10

    return clamp_score(score)


def evaluate_image(
    image_path: str | Path,
    reference_image_path: str | Path | None = None,
    prompt: str | None = None,
    evaluation_profile: str = "portfolio",
    use_clip: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
    penalize_radial_artifacts: bool = False,
) -> dict[str, Any]:
    image = resize_for_eval(load_rgb_image(image_path), size=256)
    arr = image_to_array(image)

    brightness_score = compute_brightness_score(arr)
    contrast_score = compute_contrast_score(arr)
    color_score = compute_color_score(arr)
    sharpness_score = compute_sharpness_score(arr)
    center_focus_score = compute_center_focus_score(arr)
    clutter_penalty = compute_clutter_penalty(arr)
    radial_artifact_penalty = compute_radial_artifact_penalty(arr) if penalize_radial_artifacts else 0.0

    reference_similarity = compute_pixel_histogram_reference_similarity(
        image_path=image_path,
        reference_image_path=reference_image_path,
    )

    visual_quality_score = clamp_score(
        brightness_score * 0.18
        + contrast_score * 0.20
        + color_score * 0.14
        + sharpness_score * 0.20
        + center_focus_score * 0.28
        - clutter_penalty * 0.35
        - radial_artifact_penalty * 0.45
    )

    prompt_alignment_score = None
    clip_reference_similarity = None
    clip_error = None

    if use_clip:
        try:
            prompt_alignment_score = compute_clip_prompt_alignment(
                image_path=image_path,
                prompt=prompt,
                clip_model_id=clip_model_id,
            )

            clip_reference_similarity = compute_clip_reference_similarity(
                image_path=image_path,
                reference_image_path=reference_image_path,
                clip_model_id=clip_model_id,
            )
        except Exception as exc:
            clip_error = str(exc)

    final_score = combine_scores(
        visual_quality_score=visual_quality_score,
        prompt_alignment_score=prompt_alignment_score,
        reference_similarity=reference_similarity,
        clip_reference_similarity=clip_reference_similarity,
        evaluation_profile=evaluation_profile,
    )

    return {
        "final_score": round(final_score, 2),
        "visual_quality_score": round(visual_quality_score, 2),
        "brightness_score": round(brightness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "color_score": round(color_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "center_focus_score": round(center_focus_score, 2),
        "clutter_penalty": round(clutter_penalty, 2),
        "radial_artifact_penalty": round(radial_artifact_penalty, 2),
        "reference_similarity": None if reference_similarity is None else round(reference_similarity, 2),
        "prompt_alignment_score": None if prompt_alignment_score is None else round(prompt_alignment_score, 2),
        "clip_reference_similarity": None if clip_reference_similarity is None else round(clip_reference_similarity, 2),
        "evaluation_profile": evaluation_profile,
        "clip_enabled": use_clip,
        "clip_error": clip_error,
    }


def rank_images(
    image_paths: list[str | Path],
    reference_image_path: str | Path | None = None,
    prompt: str | None = None,
    evaluation_profile: str = "portfolio",
    use_clip: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
    penalize_radial_artifacts: bool = False,
) -> list[dict[str, Any]]:
    ranked = []

    for image_path in image_paths:
        score_data = evaluate_image(
            image_path=image_path,
            reference_image_path=reference_image_path,
            prompt=prompt,
            evaluation_profile=evaluation_profile,
            use_clip=use_clip,
            clip_model_id=clip_model_id,
            penalize_radial_artifacts=penalize_radial_artifacts,
        )
        ranked.append(
            {
                "image_path": str(image_path),
                **score_data,
            }
        )

    return sorted(ranked, key=lambda item: item["final_score"], reverse=True)


def attach_scores_to_metadata(
    image_path: str | Path,
    scores: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    image_path = Path(image_path)
    metadata_path = image_path.with_suffix(".json")

    metadata: dict[str, Any] = {}

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    metadata["evaluation"] = scores

    if extra_metadata:
        metadata["refinement"] = extra_metadata

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
