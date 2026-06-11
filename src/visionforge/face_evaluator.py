from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def clamp_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def load_cv2():
    try:
        import cv2

        return cv2, None
    except Exception as exc:
        return None, str(exc)


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def pil_to_cv2_gray(image: Image.Image) -> np.ndarray:
    cv2, error = load_cv2()

    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {error}")

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray


def detect_faces(image_path: str | Path) -> list[dict[str, Any]]:
    cv2, error = load_cv2()

    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {error}")

    image = load_rgb_image(image_path)
    gray = pil_to_cv2_gray(image)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade for frontal face detection.")

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    results = []

    for x, y, w, h in faces:
        area = int(w * h)
        results.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": area,
            }
        )

    results.sort(key=lambda item: item["area"], reverse=True)
    return results


def largest_face(image_path: str | Path) -> dict[str, Any] | None:
    faces = detect_faces(image_path)
    return faces[0] if faces else None


def expand_box(
    box: dict[str, Any],
    image_width: int,
    image_height: int,
    padding: float = 0.22,
) -> tuple[int, int, int, int]:
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + w + pad_x)
    bottom = min(image_height, y + h + pad_y)

    return left, top, right, bottom


def crop_face(image_path: str | Path, box: dict[str, Any], size: int = 160) -> Image.Image:
    image = load_rgb_image(image_path)
    left, top, right, bottom = expand_box(box, image.width, image.height)
    crop = image.crop((left, top, right, bottom))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def face_center_score(box: dict[str, Any], image_width: int, image_height: int) -> float:
    face_cx = box["x"] + box["w"] / 2
    face_cy = box["y"] + box["h"] / 2

    image_cx = image_width / 2
    image_cy = image_height / 2

    distance = ((face_cx - image_cx) ** 2 + (face_cy - image_cy) ** 2) ** 0.5
    max_distance = ((image_width / 2) ** 2 + (image_height / 2) ** 2) ** 0.5

    return clamp_score(100.0 * (1.0 - distance / max_distance))


def face_size_score(box: dict[str, Any], image_width: int, image_height: int) -> float:
    face_area_ratio = (box["w"] * box["h"]) / max(1, image_width * image_height)

    # For portraits, a useful detected face area often falls around 12% to 45%.
    if face_area_ratio < 0.06:
        return clamp_score(face_area_ratio / 0.06 * 45.0)

    if face_area_ratio <= 0.38:
        return clamp_score(55.0 + face_area_ratio / 0.38 * 45.0)

    return clamp_score(100.0 - (face_area_ratio - 0.38) * 130.0)


def face_brightness_score(face_crop: Image.Image) -> float:
    arr = np.asarray(face_crop).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    mean_brightness = float(gray.mean())
    return clamp_score(100.0 - abs(mean_brightness - 0.52) * 210.0)


def face_contrast_score(face_crop: Image.Image) -> float:
    arr = np.asarray(face_crop).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    contrast = float(gray.std())

    if contrast < 0.04:
        return clamp_score(contrast / 0.04 * 45.0)

    if contrast <= 0.22:
        return clamp_score(55.0 + contrast / 0.22 * 45.0)

    return clamp_score(100.0 - (contrast - 0.22) * 140.0)


def face_sharpness_score(face_crop: Image.Image) -> float:
    cv2, error = load_cv2()

    if cv2 is None:
        return 0.0

    gray = pil_to_cv2_gray(face_crop)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Practical heuristic calibration.
    if variance < 20:
        return clamp_score(variance / 20 * 35.0)

    if variance <= 180:
        return clamp_score(35.0 + variance / 180 * 65.0)

    return 100.0


def compute_face_quality_score(
    image_path: str | Path,
    face_box: dict[str, Any],
) -> float:
    image = load_rgb_image(image_path)
    crop = crop_face(image_path, face_box)

    center = face_center_score(face_box, image.width, image.height)
    size = face_size_score(face_box, image.width, image.height)
    brightness = face_brightness_score(crop)
    contrast = face_contrast_score(crop)
    sharpness = face_sharpness_score(crop)

    score = (
        center * 0.18
        + size * 0.18
        + brightness * 0.18
        + contrast * 0.18
        + sharpness * 0.28
    )

    return clamp_score(score)


def histogram_overlap(arr_a: np.ndarray, arr_b: np.ndarray, bins: int = 32) -> float:
    scores = []

    for channel in range(arr_a.shape[2]):
        hist_a, _ = np.histogram(arr_a[:, :, channel], bins=bins, range=(0, 1), density=True)
        hist_b, _ = np.histogram(arr_b[:, :, channel], bins=bins, range=(0, 1), density=True)

        hist_a = hist_a / (hist_a.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)

        scores.append(float(np.minimum(hist_a, hist_b).sum() * 100.0))

    return float(np.mean(scores))


def grayscale_structure_similarity(crop_a: Image.Image, crop_b: Image.Image) -> float:
    gray_a = np.asarray(crop_a.convert("L")).astype(np.float32) / 255.0
    gray_b = np.asarray(crop_b.convert("L")).astype(np.float32) / 255.0

    mae = float(np.abs(gray_a - gray_b).mean())
    return clamp_score(100.0 * (1.0 - mae))


def box_similarity(
    box_a: dict[str, Any],
    image_a_size: tuple[int, int],
    box_b: dict[str, Any],
    image_b_size: tuple[int, int],
) -> float:
    width_a, height_a = image_a_size
    width_b, height_b = image_b_size

    ax = (box_a["x"] + box_a["w"] / 2) / max(1, width_a)
    ay = (box_a["y"] + box_a["h"] / 2) / max(1, height_a)
    aw = box_a["w"] / max(1, width_a)
    ah = box_a["h"] / max(1, height_a)

    bx = (box_b["x"] + box_b["w"] / 2) / max(1, width_b)
    by = (box_b["y"] + box_b["h"] / 2) / max(1, height_b)
    bw = box_b["w"] / max(1, width_b)
    bh = box_b["h"] / max(1, height_b)

    center_distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    size_distance = ((aw - bw) ** 2 + (ah - bh) ** 2) ** 0.5

    score = 100.0 - center_distance * 180.0 - size_distance * 160.0
    return clamp_score(score)


def compute_face_reference_similarity(
    image_path: str | Path,
    reference_image_path: str | Path,
    image_face: dict[str, Any],
    reference_face: dict[str, Any],
) -> float:
    image = load_rgb_image(image_path)
    reference = load_rgb_image(reference_image_path)

    crop_a = crop_face(image_path, image_face, size=160)
    crop_b = crop_face(reference_image_path, reference_face, size=160)

    arr_a = np.asarray(crop_a).astype(np.float32) / 255.0
    arr_b = np.asarray(crop_b).astype(np.float32) / 255.0

    color_similarity = histogram_overlap(arr_a, arr_b)
    structure_similarity = grayscale_structure_similarity(crop_a, crop_b)
    geometry_similarity = box_similarity(
        image_face,
        (image.width, image.height),
        reference_face,
        (reference.width, reference.height),
    )

    score = (
        color_similarity * 0.30
        + structure_similarity * 0.42
        + geometry_similarity * 0.28
    )

    return clamp_score(score)


def evaluate_faces(
    image_path: str | Path,
    reference_image_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        image_faces = detect_faces(image_path)
        image_face = image_faces[0] if image_faces else None

        face_quality_score = None
        face_reference_similarity = None
        reference_face_count = None

        if image_face is not None:
            face_quality_score = compute_face_quality_score(image_path, image_face)

        if reference_image_path:
            reference_faces = detect_faces(reference_image_path)
            reference_face = reference_faces[0] if reference_faces else None
            reference_face_count = len(reference_faces)

            if image_face is not None and reference_face is not None:
                face_reference_similarity = compute_face_reference_similarity(
                    image_path=image_path,
                    reference_image_path=reference_image_path,
                    image_face=image_face,
                    reference_face=reference_face,
                )

        return {
            "face_evaluator_available": True,
            "face_error": None,
            "face_count": len(image_faces),
            "main_face_box": image_face,
            "face_quality_score": None if face_quality_score is None else round(face_quality_score, 2),
            "reference_face_count": reference_face_count,
            "face_reference_similarity": None
            if face_reference_similarity is None
            else round(face_reference_similarity, 2),
        }

    except Exception as exc:
        return {
            "face_evaluator_available": False,
            "face_error": str(exc),
            "face_count": None,
            "main_face_box": None,
            "face_quality_score": None,
            "reference_face_count": None,
            "face_reference_similarity": None,
        }
