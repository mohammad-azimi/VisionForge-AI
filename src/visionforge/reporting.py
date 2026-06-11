from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


@dataclass
class OutputRecord:
    image_path: str
    metadata_path: str
    filename: str
    project_name: str
    generation_mode: str
    model_id: str
    seed: str
    final_score: float | None
    visual_quality_score: float | None
    prompt_alignment_score: float | None
    reference_similarity: float | None
    clip_reference_similarity: float | None
    face_quality_score: float | None
    face_reference_similarity: float | None
    face_count: int | None
    session_id: str
    iteration: int | None
    prompt_label: str
    generation_kind: str
    parent_source: str
    prompt: str
    negative_prompt: str
    raw_metadata: dict[str, Any]


def safe_get(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key, default)

    return current


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None

        return float(value)
    except (TypeError, ValueError):
        return None


def to_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None

        return int(value)
    except (TypeError, ValueError):
        return None


def find_image_for_metadata(metadata_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = metadata_path.with_suffix(extension)
        if candidate.exists():
            return candidate

    return None


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_prompt(metadata: dict[str, Any]) -> str:
    return (
        safe_get(metadata, "prompt", default="")
        or safe_get(metadata, "prompt_used", default="")
        or safe_get(metadata, "refinement", "prompt_used", default="")
        or safe_get(metadata, "generation", "prompt", default="")
    )


def extract_negative_prompt(metadata: dict[str, Any]) -> str:
    return (
        safe_get(metadata, "negative_prompt", default="")
        or safe_get(metadata, "negative_prompt_used", default="")
        or safe_get(metadata, "refinement", "negative_prompt_used", default="")
        or safe_get(metadata, "generation", "negative_prompt", default="")
    )


def extract_project_name(metadata: dict[str, Any], image_path: Path) -> str:
    return (
        safe_get(metadata, "project_name", default="")
        or safe_get(metadata, "config", "project_name", default="")
        or safe_get(metadata, "generation", "project_name", default="")
        or image_path.stem.split("_")[0]
    )


def extract_generation_mode(metadata: dict[str, Any]) -> str:
    return (
        safe_get(metadata, "generation_mode", default="")
        or safe_get(metadata, "config", "generation_mode", default="")
        or safe_get(metadata, "refinement", "generation_kind", default="")
        or "unknown"
    )


def extract_model_id(metadata: dict[str, Any]) -> str:
    return (
        safe_get(metadata, "model_id", default="")
        or safe_get(metadata, "config", "model_id", default="")
        or safe_get(metadata, "generation", "model_id", default="")
    )


def extract_seed(metadata: dict[str, Any]) -> str:
    value = (
        safe_get(metadata, "seed", default="")
        or safe_get(metadata, "config", "seed", default="")
        or safe_get(metadata, "generation", "seed", default="")
    )

    return str(value) if value != "" else ""


def record_from_metadata(metadata_path: Path) -> OutputRecord | None:
    image_path = find_image_for_metadata(metadata_path)

    if image_path is None:
        return None

    metadata = load_json(metadata_path)
    evaluation = metadata.get("evaluation", {}) if isinstance(metadata.get("evaluation"), dict) else {}
    refinement = metadata.get("refinement", {}) if isinstance(metadata.get("refinement"), dict) else {}

    return OutputRecord(
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        filename=image_path.name,
        project_name=str(extract_project_name(metadata, image_path)),
        generation_mode=str(extract_generation_mode(metadata)),
        model_id=str(extract_model_id(metadata)),
        seed=extract_seed(metadata),
        final_score=to_float_or_none(evaluation.get("final_score")),
        visual_quality_score=to_float_or_none(evaluation.get("visual_quality_score")),
        prompt_alignment_score=to_float_or_none(evaluation.get("prompt_alignment_score")),
        reference_similarity=to_float_or_none(evaluation.get("reference_similarity")),
        clip_reference_similarity=to_float_or_none(evaluation.get("clip_reference_similarity")),
        face_quality_score=to_float_or_none(evaluation.get("face_quality_score")),
        face_reference_similarity=to_float_or_none(evaluation.get("face_reference_similarity")),
        face_count=to_int_or_none(evaluation.get("face_count")),
        session_id=str(refinement.get("session_id", "")),
        iteration=to_int_or_none(refinement.get("iteration")),
        prompt_label=str(refinement.get("prompt_label", "")),
        generation_kind=str(refinement.get("generation_kind", "")),
        parent_source=str(refinement.get("parent_source", "")),
        prompt=str(extract_prompt(metadata)),
        negative_prompt=str(extract_negative_prompt(metadata)),
        raw_metadata=metadata,
    )


def load_output_records(output_dir: str | Path = "outputs") -> list[OutputRecord]:
    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    records: list[OutputRecord] = []

    for metadata_path in sorted(output_path.glob("*.json")):
        record = record_from_metadata(metadata_path)

        if record is not None:
            records.append(record)

    return records


def record_to_dict(record: OutputRecord) -> dict[str, Any]:
    return {
        "filename": record.filename,
        "project_name": record.project_name,
        "generation_mode": record.generation_mode,
        "model_id": record.model_id,
        "seed": record.seed,
        "final_score": record.final_score,
        "visual_quality_score": record.visual_quality_score,
        "prompt_alignment_score": record.prompt_alignment_score,
        "reference_similarity": record.reference_similarity,
        "clip_reference_similarity": record.clip_reference_similarity,
        "face_quality_score": record.face_quality_score,
        "face_reference_similarity": record.face_reference_similarity,
        "face_count": record.face_count,
        "session_id": record.session_id,
        "iteration": record.iteration,
        "prompt_label": record.prompt_label,
        "generation_kind": record.generation_kind,
        "parent_source": record.parent_source,
        "image_path": record.image_path,
    }


def sort_records_by_score(records: list[OutputRecord]) -> list[OutputRecord]:
    return sorted(
        records,
        key=lambda record: record.final_score if record.final_score is not None else -1.0,
        reverse=True,
    )


def filter_records(
    records: list[OutputRecord],
    project_name: str | None = None,
    session_id: str | None = None,
    only_scored: bool = False,
    only_faces: bool = False,
) -> list[OutputRecord]:
    filtered = records

    if project_name and project_name != "All":
        filtered = [record for record in filtered if record.project_name == project_name]

    if session_id and session_id != "All":
        filtered = [record for record in filtered if record.session_id == session_id]

    if only_scored:
        filtered = [record for record in filtered if record.final_score is not None]

    if only_faces:
        filtered = [
            record
            for record in filtered
            if record.face_count is not None and record.face_count > 0
        ]

    return filtered


def build_markdown_report(records: list[OutputRecord], title: str = "VisionForge-AI Experiment Report") -> str:
    sorted_records = sort_records_by_score(records)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("Generated by VisionForge-AI.")
    lines.append("")
    lines.append(f"Total outputs: {len(sorted_records)}")
    lines.append("")

    if not sorted_records:
        lines.append("No outputs found.")
        return "\n".join(lines)

    best = sorted_records[0]

    lines.append("## Best Output")
    lines.append("")
    lines.append(f"- File: `{best.filename}`")
    lines.append(f"- Final score: `{best.final_score}`")
    lines.append(f"- Visual quality: `{best.visual_quality_score}`")
    lines.append(f"- Prompt alignment: `{best.prompt_alignment_score}`")
    lines.append(f"- Reference similarity: `{best.reference_similarity}`")
    lines.append(f"- CLIP reference similarity: `{best.clip_reference_similarity}`")
    lines.append(f"- Face quality: `{best.face_quality_score}`")
    lines.append(f"- Face reference similarity: `{best.face_reference_similarity}`")
    lines.append(f"- Session ID: `{best.session_id}`")
    lines.append(f"- Iteration: `{best.iteration}`")
    lines.append(f"- Prompt label: `{best.prompt_label}`")
    lines.append(f"- Generation kind: `{best.generation_kind}`")
    lines.append("")
    lines.append(f"![Best output]({best.image_path})")
    lines.append("")

    lines.append("## Top Outputs")
    lines.append("")

    for index, record in enumerate(sorted_records[:10], start=1):
        lines.append(f"### {index}. {record.filename}")
        lines.append("")
        lines.append(f"- Final score: `{record.final_score}`")
        lines.append(f"- Visual quality: `{record.visual_quality_score}`")
        lines.append(f"- Face quality: `{record.face_quality_score}`")
        lines.append(f"- Face reference similarity: `{record.face_reference_similarity}`")
        lines.append(f"- Prompt label: `{record.prompt_label}`")
        lines.append(f"- Generation kind: `{record.generation_kind}`")
        lines.append(f"- Parent source: `{record.parent_source}`")
        lines.append("")
        lines.append(f"![Output {index}]({record.image_path})")
        lines.append("")

        if record.prompt:
            lines.append("Prompt:")
            lines.append("")
            lines.append("```text")
            lines.append(record.prompt)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def build_json_report(records: list[OutputRecord]) -> str:
    data = [record_to_dict(record) for record in sort_records_by_score(records)]

    return json.dumps(
        {
            "total_outputs": len(data),
            "outputs": data,
        },
        indent=2,
        ensure_ascii=False,
    )