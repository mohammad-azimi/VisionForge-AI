from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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

    # Best range is neither too dark nor too bright.
    score = 100.0 - abs(mean_brightness - 0.50) * 220.0
    return clamp_score(score)


def compute_contrast_score(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    contrast = float(gray.std())

    # Very low contrast is bad. Extremely high contrast can also mean harsh/noisy output.
    if contrast < 0.03:
        return clamp_score(contrast / 0.03 * 30.0)

    if contrast <= 0.23:
        return clamp_score(45.0 + contrast / 0.23 * 55.0)

    return clamp_score(100.0 - (contrast - 0.23) * 160.0)


def compute_color_score(arr: np.ndarray) -> float:
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = float((max_channel - min_channel).mean())

    # A small-to-medium amount of color usually works well for clean portfolio visuals.
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

    # Too much high-frequency detail may indicate noise, artifacts, or clutter.
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


def compute_reference_similarity(
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

    # Color histogram similarity gives a softer similarity signal.
    hist_scores = []
    for channel in range(3):
        hist_a, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(ref[:, :, channel], bins=32, range=(0.0, 1.0), density=True)

        hist_a = hist_a / (hist_a.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)

        overlap = np.minimum(hist_a, hist_b).sum()
        hist_scores.append(float(overlap * 100.0))

    histogram_similarity = float(np.mean(hist_scores))

    return clamp_score(pixel_similarity * 0.65 + histogram_similarity * 0.35)


def evaluate_image(
    image_path: str | Path,
    reference_image_path: str | Path | None = None,
) -> dict[str, Any]:
    image = resize_for_eval(load_rgb_image(image_path), size=256)
    arr = image_to_array(image)

    brightness_score = compute_brightness_score(arr)
    contrast_score = compute_contrast_score(arr)
    color_score = compute_color_score(arr)
    sharpness_score = compute_sharpness_score(arr)
    center_focus_score = compute_center_focus_score(arr)
    clutter_penalty = compute_clutter_penalty(arr)
    reference_similarity = compute_reference_similarity(image_path, reference_image_path)

    visual_quality_score = clamp_score(
        brightness_score * 0.18
        + contrast_score * 0.22
        + color_score * 0.16
        + sharpness_score * 0.22
        + center_focus_score * 0.22
        - clutter_penalty * 0.35
    )

    if reference_similarity is None:
        final_score = visual_quality_score
    else:
        final_score = clamp_score(visual_quality_score * 0.45 + reference_similarity * 0.55)

    return {
        "final_score": round(final_score, 2),
        "visual_quality_score": round(visual_quality_score, 2),
        "brightness_score": round(brightness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "color_score": round(color_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "center_focus_score": round(center_focus_score, 2),
        "clutter_penalty": round(clutter_penalty, 2),
        "reference_similarity": None if reference_similarity is None else round(reference_similarity, 2),
    }


def rank_images(
    image_paths: list[str | Path],
    reference_image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    ranked = []

    for image_path in image_paths:
        score_data = evaluate_image(
            image_path=image_path,
            reference_image_path=reference_image_path,
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
