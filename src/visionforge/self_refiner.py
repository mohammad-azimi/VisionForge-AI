from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .evaluator import DEFAULT_CLIP_MODEL_ID, attach_scores_to_metadata, rank_images
from .generator import (
    GenerationConfig,
    generate_image,
    generate_image_from_image,
)
from .prompt_mutator import PromptVariant, build_prompt_variants


RADIAL_ESCAPE_NEGATIVE_PROMPT = (
    "eye, iris, pupil, eyeball, portal, vortex, tunnel, target symbol, "
    "concentric circles, repeated rings, circular artifact, ring pattern, "
    "hypnotic circle, abstract circular texture"
)


@dataclass
class RefinementConfig:
    prompt: str
    negative_prompt: str
    model_id: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    output_dir: str
    project_name: str
    output_type: str
    style: str
    iterations: int = 3
    candidates_per_iteration: int = 2
    strength_start: float = 0.42
    strength_decay: float = 0.07
    min_strength: float = 0.20
    initial_image_path: str | None = None
    reference_image_path: str | None = None
    evaluation_profile: str = "portfolio"
    use_clip: bool = False
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    penalize_radial_artifacts: bool = False

    exploration_candidates_per_iteration: int = 1
    artifact_escape_threshold: float = 18.0
    restart_when_all_candidates_have_artifacts: bool = True

    enable_prompt_mutation: bool = True


def build_generation_config(config: RefinementConfig, seed: int, mode: str) -> GenerationConfig:
    return GenerationConfig(
        model_id=config.model_id,
        width=config.width,
        height=config.height,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=seed,
        output_dir=config.output_dir,
        project_name=config.project_name,
        output_type=config.output_type,
        style=config.style,
        generation_mode=mode,
    )


def strength_for_iteration(config: RefinementConfig, iteration_index: int) -> float:
    strength = config.strength_start - (iteration_index * config.strength_decay)
    return max(config.min_strength, strength)


def effective_negative_prompt(config: RefinementConfig) -> str:
    if not config.penalize_radial_artifacts:
        return config.negative_prompt

    if RADIAL_ESCAPE_NEGATIVE_PROMPT.lower() in config.negative_prompt.lower():
        return config.negative_prompt

    if not config.negative_prompt.strip():
        return RADIAL_ESCAPE_NEGATIVE_PROMPT

    return f"{config.negative_prompt}, {RADIAL_ESCAPE_NEGATIVE_PROMPT}"


def build_iteration_prompt_variants(
    config: RefinementConfig,
    candidate_count: int,
) -> list[PromptVariant]:
    if not config.enable_prompt_mutation:
        return [
            PromptVariant(
                label="base",
                prompt=config.prompt,
                negative_prompt=effective_negative_prompt(config),
            )
        ]

    requested_count = max(candidate_count + 2, 5)

    return build_prompt_variants(
        prompt=config.prompt,
        negative_prompt=effective_negative_prompt(config),
        evaluation_profile=config.evaluation_profile,
        num_variants=requested_count,
        avoid_radial_artifacts=config.penalize_radial_artifacts,
        include_base=True,
    )


def choose_prompt_variant(
    prompt_variants: list[PromptVariant],
    candidate_index: int,
    is_fresh_candidate: bool,
) -> PromptVariant:
    if not prompt_variants:
        return PromptVariant(label="base", prompt="", negative_prompt="")

    if len(prompt_variants) == 1:
        return prompt_variants[0]

    if is_fresh_candidate:
        # Exploration candidates skip the base prompt when possible.
        index = 1 + (candidate_index % (len(prompt_variants) - 1))
        return prompt_variants[index]

    # Refinement candidates can still use base + mild diversity.
    index = candidate_index % len(prompt_variants)
    return prompt_variants[index]


def is_image_file_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False

    path = Path(value)
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def extract_image_path(output: Any) -> str:
    if isinstance(output, dict):
        preferred_keys = [
            "image_path",
            "path",
            "output_path",
            "saved_image_path",
            "file_path",
            "filename",
        ]

        for key in preferred_keys:
            value = output.get(key)
            if is_image_file_path(value):
                return str(value)

        for value in output.values():
            try:
                return extract_image_path(value)
            except (TypeError, ValueError):
                continue

    if is_image_file_path(output):
        return str(output)

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, Image.Image):
                continue

            try:
                return extract_image_path(item)
            except (TypeError, ValueError):
                continue

    raise ValueError(
        "Could not extract image path from generator output. "
        f"Output type: {type(output)} | Output value: {output}"
    )


def function_accepts_kwargs(function: Any) -> bool:
    signature = inspect.signature(function)
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def filter_kwargs_for_function(function: Any, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)

    if function_accepts_kwargs(function):
        return values

    return {
        name: value
        for name, value in values.items()
        if name in signature.parameters
    }


def call_text_to_image_generator(
    prompt: str,
    negative_prompt: str,
    generation_config: GenerationConfig,
) -> Any:
    values = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "negative": negative_prompt,
        "config": generation_config,
        "generation_config": generation_config,
    }

    kwargs = filter_kwargs_for_function(generate_image, values)

    try:
        return generate_image(**kwargs)
    except TypeError:
        try:
            return generate_image(prompt, negative_prompt, generation_config)
        except TypeError as exc:
            signature = inspect.signature(generate_image)
            raise TypeError(
                "Could not call generate_image with the supported signatures. "
                f"Current signature: {signature}"
            ) from exc


def call_image_to_image_generator(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    generation_config: GenerationConfig,
    strength: float,
) -> Any:
    image_path = str(image_path)
    pil_image = Image.open(image_path).convert("RGB")

    values = {
        "init_image_path": image_path,
        "image_path": image_path,
        "input_image_path": image_path,
        "source_image_path": image_path,
        "reference_image_path": image_path,
        "path": image_path,
        "init_image": pil_image,
        "image": pil_image,
        "input_image": pil_image,
        "source_image": pil_image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "negative": negative_prompt,
        "config": generation_config,
        "generation_config": generation_config,
        "strength": strength,
        "img2img_strength": strength,
    }

    kwargs = filter_kwargs_for_function(generate_image_from_image, values)

    try:
        return generate_image_from_image(**kwargs)
    except TypeError:
        fallback_calls = [
            lambda: generate_image_from_image(image_path, prompt, negative_prompt, generation_config, strength),
            lambda: generate_image_from_image(prompt, negative_prompt, generation_config, image_path, strength),
            lambda: generate_image_from_image(pil_image, prompt, negative_prompt, generation_config, strength),
            lambda: generate_image_from_image(prompt, negative_prompt, generation_config, pil_image, strength),
            lambda: generate_image_from_image(prompt, negative_prompt, pil_image, generation_config, strength),
        ]

        last_error: Exception | None = None

        for fallback_call in fallback_calls:
            try:
                return fallback_call()
            except TypeError as exc:
                last_error = exc
                continue

        signature = inspect.signature(generate_image_from_image)
        raise TypeError(
            "Could not call generate_image_from_image with the supported signatures. "
            f"Current signature: {signature}"
        ) from last_error


def should_make_fresh_candidate(
    config: RefinementConfig,
    iteration_index: int,
    candidate_index: int,
    parent_image_path: str | None,
) -> bool:
    if parent_image_path is None:
        return True

    if config.reference_image_path:
        return False

    if iteration_index == 0:
        return False

    exploration_count = max(0, config.exploration_candidates_per_iteration)
    return candidate_index >= max(1, config.candidates_per_iteration - exploration_count)


def generate_candidates_for_iteration(
    config: RefinementConfig,
    iteration_index: int,
    parent_image_path: str | None,
) -> list[dict[str, Any]]:
    generated_records: list[dict[str, Any]] = []
    candidate_count = max(1, config.candidates_per_iteration)
    prompt_variants = build_iteration_prompt_variants(config, candidate_count)

    for candidate_index in range(candidate_count):
        seed = config.seed + iteration_index * 1000 + candidate_index

        make_fresh_candidate = should_make_fresh_candidate(
            config=config,
            iteration_index=iteration_index,
            candidate_index=candidate_index,
            parent_image_path=parent_image_path,
        )

        prompt_variant = choose_prompt_variant(
            prompt_variants=prompt_variants,
            candidate_index=candidate_index,
            is_fresh_candidate=make_fresh_candidate or parent_image_path is None,
        )

        if parent_image_path and not make_fresh_candidate:
            generation_config = build_generation_config(
                config=config,
                seed=seed,
                mode="self-refine-image-to-image",
            )

            output = call_image_to_image_generator(
                image_path=parent_image_path,
                prompt=prompt_variant.prompt,
                negative_prompt=prompt_variant.negative_prompt,
                generation_config=generation_config,
                strength=strength_for_iteration(config, iteration_index),
            )

            generation_kind = "image-to-image-refinement"
            candidate_strength = strength_for_iteration(config, iteration_index)
        else:
            generation_config = build_generation_config(
                config=config,
                seed=seed,
                mode="self-refine-text-to-image-exploration",
            )

            output = call_text_to_image_generator(
                prompt=prompt_variant.prompt,
                negative_prompt=prompt_variant.negative_prompt,
                generation_config=generation_config,
            )

            generation_kind = "text-to-image-exploration"
            candidate_strength = None

        image_path = extract_image_path(output)

        generated_records.append(
            {
                "image_path": image_path,
                "prompt": prompt_variant.prompt,
                "negative_prompt": prompt_variant.negative_prompt,
                "prompt_label": prompt_variant.label,
                "generation_kind": generation_kind,
                "strength": candidate_strength,
            }
        )

    return generated_records


def enrich_ranked_candidates(
    ranked_candidates: list[dict[str, Any]],
    generated_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_map = {record["image_path"]: record for record in generated_records}
    enriched: list[dict[str, Any]] = []

    for candidate in ranked_candidates:
        extra = record_map.get(candidate["image_path"], {})
        merged = {**candidate, **extra}
        enriched.append(merged)

    return enriched


def select_next_parent(
    ranked_candidates: list[dict[str, Any]],
    config: RefinementConfig,
) -> tuple[dict[str, Any], str | None, str]:
    best_candidate = ranked_candidates[0]

    if not config.penalize_radial_artifacts:
        return best_candidate, best_candidate["image_path"], "best_score"

    clean_candidates = [
        candidate
        for candidate in ranked_candidates
        if float(candidate.get("radial_artifact_penalty") or 0.0) < config.artifact_escape_threshold
    ]

    if clean_candidates:
        selected = clean_candidates[0]
        return selected, selected["image_path"], "best_clean_candidate"

    if config.restart_when_all_candidates_have_artifacts:
        return best_candidate, None, "restart_due_to_radial_artifacts"

    return best_candidate, best_candidate["image_path"], "best_score_with_artifact_warning"


def run_self_refinement(config: RefinementConfig) -> dict[str, Any]:
    session_id = datetime.now().strftime("refine_%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_overall: dict[str, Any] | None = None
    parent_image_path = config.initial_image_path

    for iteration_index in range(max(1, config.iterations)):
        generated_records = generate_candidates_for_iteration(
            config=config,
            iteration_index=iteration_index,
            parent_image_path=parent_image_path,
        )

        ranked_candidates = rank_images(
            image_paths=[record["image_path"] for record in generated_records],
            reference_image_path=config.reference_image_path,
            prompt=config.prompt,
            evaluation_profile=config.evaluation_profile,
            use_clip=config.use_clip,
            clip_model_id=config.clip_model_id,
            penalize_radial_artifacts=config.penalize_radial_artifacts,
        )

        ranked_candidates = enrich_ranked_candidates(
            ranked_candidates=ranked_candidates,
            generated_records=generated_records,
        )

        selected_candidate, next_parent_image_path, selection_reason = select_next_parent(
            ranked_candidates=ranked_candidates,
            config=config,
        )

        for rank_index, candidate in enumerate(ranked_candidates, start=1):
            refinement_metadata = {
                "session_id": session_id,
                "iteration": iteration_index + 1,
                "rank_in_iteration": rank_index,
                "parent_image_path": parent_image_path,
                "next_parent_image_path": next_parent_image_path,
                "selection_reason": selection_reason,
                "reference_image_path": config.reference_image_path,
                "strength": candidate.get("strength"),
                "evaluation_profile": config.evaluation_profile,
                "use_clip": config.use_clip,
                "clip_model_id": config.clip_model_id,
                "penalize_radial_artifacts": config.penalize_radial_artifacts,
                "exploration_candidates_per_iteration": config.exploration_candidates_per_iteration,
                "artifact_escape_threshold": config.artifact_escape_threshold,
                "enable_prompt_mutation": config.enable_prompt_mutation,
                "prompt_label": candidate.get("prompt_label"),
                "generation_kind": candidate.get("generation_kind"),
                "prompt_used": candidate.get("prompt"),
                "negative_prompt_used": candidate.get("negative_prompt"),
            }

            attach_scores_to_metadata(
                image_path=candidate["image_path"],
                scores={
                    key: value
                    for key, value in candidate.items()
                    if key != "image_path"
                },
                extra_metadata=refinement_metadata,
            )

        parent_image_path = next_parent_image_path

        if best_overall is None or selected_candidate["final_score"] > best_overall["final_score"]:
            best_overall = selected_candidate

        history.append(
            {
                "iteration": iteration_index + 1,
                "best_image_path": selected_candidate["image_path"],
                "best_score": selected_candidate["final_score"],
                "visual_quality_score": selected_candidate["visual_quality_score"],
                "prompt_alignment_score": selected_candidate.get("prompt_alignment_score"),
                "reference_similarity": selected_candidate.get("reference_similarity"),
                "clip_reference_similarity": selected_candidate.get("clip_reference_similarity"),
                "selection_reason": selection_reason,
                "next_parent_image_path": next_parent_image_path,
                "selected_prompt_label": selected_candidate.get("prompt_label"),
                "selected_generation_kind": selected_candidate.get("generation_kind"),
                "candidates": ranked_candidates,
            }
        )

    return {
        "session_id": session_id,
        "best_image_path": best_overall["image_path"] if best_overall else None,
        "best_score": best_overall["final_score"] if best_overall else None,
        "history": history,
    }