from pathlib import Path
import json


def load_metadata(metadata_path: Path) -> dict:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_generation_records(output_dir: str = "outputs") -> list[dict]:
    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    records = []

    for image_path in output_path.glob("*.png"):
        metadata_path = image_path.with_suffix(".json")
        metadata = load_metadata(metadata_path) if metadata_path.exists() else {}

        records.append(
            {
                "image_path": image_path,
                "metadata_path": metadata_path if metadata_path.exists() else None,
                "metadata": metadata,
                "created_at": image_path.stat().st_mtime,
                "prompt": metadata.get("prompt", ""),
                "negative_prompt": metadata.get("negative_prompt", ""),
                "model_id": metadata.get("model_id", "unknown"),
                "seed": metadata.get("seed", "unknown"),
                "width": metadata.get("width", "unknown"),
                "height": metadata.get("height", "unknown"),
            }
        )

    return sorted(records, key=lambda item: item["created_at"], reverse=True)


def clear_generation_history(output_dir: str = "outputs") -> int:
    output_path = Path(output_dir)

    if not output_path.exists():
        return 0

    deleted_count = 0

    for file_path in output_path.glob("visionforge_*"):
        if file_path.suffix.lower() in {".png", ".json"}:
            file_path.unlink()
            deleted_count += 1

    return deleted_count
