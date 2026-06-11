from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.visionforge.evaluator import DEFAULT_CLIP_MODEL_ID
from src.visionforge.presets import MODEL_PRESETS, PROJECT_PRESETS, SIZE_PRESETS
from src.visionforge.prompt_builder import build_prompt_bundle
from src.visionforge.self_refiner import RefinementConfig, run_self_refinement


OUTPUT_DIR = Path("outputs")


st.set_page_config(
    page_title="Benchmark Studio | VisionForge-AI",
    page_icon="🧪",
    layout="wide",
)


EVALUATION_PROFILE_LABELS = {
    "Portfolio Cover": "portfolio",
    "General Image": "general",
    "Portrait / Face": "portrait",
    "Reference Match": "reference_match",
}


def model_options() -> dict[str, str]:
    return {
        label: model_id
        for label, model_id in MODEL_PRESETS.items()
        if label != "Custom model" and model_id
    }


def collect_best_candidates(results: list[dict]) -> list[dict]:
    rows = []

    for result in results:
        model_label = result["model_label"]
        model_id = result["model_id"]
        run_result = result["result"]

        best_candidate = None

        for item in run_result.get("history", []):
            for candidate in item.get("candidates", []):
                if best_candidate is None:
                    best_candidate = candidate
                    continue

                current_score = candidate.get("final_score") or -1
                best_score = best_candidate.get("final_score") or -1

                if current_score > best_score:
                    best_candidate = candidate

        if best_candidate:
            rows.append(
                {
                    "model_label": model_label,
                    "model_id": model_id,
                    "best_score": best_candidate.get("final_score"),
                    "visual_quality": best_candidate.get("visual_quality_score"),
                    "prompt_alignment": best_candidate.get("prompt_alignment_score"),
                    "reference_similarity": best_candidate.get("reference_similarity"),
                    "clip_reference_similarity": best_candidate.get("clip_reference_similarity"),
                    "face_quality": best_candidate.get("face_quality_score"),
                    "face_reference_similarity": best_candidate.get("face_reference_similarity"),
                    "prompt_label": best_candidate.get("prompt_label"),
                    "generation_kind": best_candidate.get("generation_kind"),
                    "image_path": best_candidate.get("image_path"),
                }
            )

    return sorted(
        rows,
        key=lambda row: row["best_score"] if row["best_score"] is not None else -1,
        reverse=True,
    )


def display_benchmark_results(results: list[dict]) -> None:
    st.subheader("Benchmark Summary")

    best_rows = collect_best_candidates(results)

    if not best_rows:
        st.warning("No benchmark results found.")
        return

    st.dataframe(best_rows, use_container_width=True)

    st.subheader("Best Output per Model")

    cols = st.columns(min(3, len(best_rows)))

    for index, row in enumerate(best_rows):
        with cols[index % len(cols)]:
            st.image(row["image_path"], use_container_width=True)
            st.caption(f"Model: {row['model_label']}")
            st.caption(f"Best score: {row['best_score']}")
            st.caption(f"Visual quality: {row['visual_quality']}")

            if row.get("prompt_alignment") is not None:
                st.caption(f"Prompt alignment: {row['prompt_alignment']}")

            if row.get("face_quality") is not None:
                st.caption(f"Face quality: {row['face_quality']}")

    st.subheader("Detailed Runs")

    for result in results:
        with st.expander(f"{result['model_label']} — {result['model_id']}"):
            run_result = result["result"]

            st.write(f"**Best score:** {run_result.get('best_score')}")
            st.write(f"**Best image:** `{run_result.get('best_image_path')}`")

            for item in run_result.get("history", []):
                st.write(f"### Iteration {item.get('iteration')}")

                candidates = item.get("candidates", [])
                cols = st.columns(min(4, max(1, len(candidates))))

                for index, candidate in enumerate(candidates):
                    with cols[index % len(cols)]:
                        st.image(candidate["image_path"], use_container_width=True)
                        st.caption(f"Score: {candidate.get('final_score')}")
                        st.caption(f"Quality: {candidate.get('visual_quality_score')}")
                        st.caption(f"Prompt label: {candidate.get('prompt_label')}")
                        st.caption(f"Kind: {candidate.get('generation_kind')}")

                        with st.expander("Prompt"):
                            st.write(candidate.get("prompt", ""))

                        with st.expander("Negative"):
                            st.write(candidate.get("negative_prompt", ""))


st.title("🧪 Benchmark Studio")
st.write(
    "Compare multiple diffusion models or presets on the same prompt, score their outputs, "
    "and identify the best model/settings for a project."
)

st.info(
    "Benchmarking can be slow because it generates images for each selected model. "
    "Start with fast settings, then run higher-quality tests."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Prompt Setup")

    selected_project = st.selectbox(
        "Project preset",
        list(PROJECT_PRESETS.keys()),
        index=0,
    )

    project_preset = PROJECT_PRESETS[selected_project]

    project_name = st.text_input(
        "Project name",
        value=project_preset["project_name"],
    )

    category = st.text_input(
        "Category",
        value=project_preset["category"],
    )

    output_type = st.text_input(
        "Output type",
        value=project_preset["output_type"],
    )

    style = st.text_input(
        "Visual style",
        value=project_preset["style"],
    )

    color_palette = st.text_input(
        "Color palette",
        value=project_preset["color_palette"],
    )

    user_prompt = st.text_area(
        "Main idea",
        value=project_preset["idea"],
        height=150,
    )

    custom_negative_prompt = st.text_area(
        "Extra negative prompt",
        value="",
        height=120,
    )

    prompt_bundle = build_prompt_bundle(
        project_name=project_name,
        category=category,
        output_type=output_type,
        style=style,
        color_palette=color_palette,
        user_prompt=user_prompt,
        custom_negative_prompt=custom_negative_prompt,
    )

    with st.expander("Final prompt"):
        st.write(prompt_bundle["prompt"])

    with st.expander("Final negative prompt"):
        st.write(prompt_bundle["negative_prompt"])

with right_col:
    st.subheader("Benchmark Settings")

    available_models = model_options()

    selected_model_labels = st.multiselect(
        "Models to compare",
        options=list(available_models.keys()),
        default=[
            label
            for label in ["SD Turbo Fast", "SD 1.5 Quality"]
            if label in available_models
        ],
    )

    selected_size = st.selectbox(
        "Size preset",
        list(SIZE_PRESETS.keys()),
        index=0,
    )

    size_data = SIZE_PRESETS[selected_size]

    width = st.number_input(
        "Width",
        min_value=256,
        max_value=1024,
        value=int(size_data["width"]),
        step=64,
    )

    height = st.number_input(
        "Height",
        min_value=256,
        max_value=1024,
        value=int(size_data["height"]),
        step=64,
    )

    seed = st.number_input(
        "Base seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    candidates_per_model = st.slider(
        "Candidates per model",
        min_value=1,
        max_value=5,
        value=2,
    )

    num_inference_steps = st.slider(
        "Inference steps",
        min_value=2,
        max_value=50,
        value=8,
    )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
    )

    st.subheader("Evaluator Settings")

    evaluation_profile_label = st.selectbox(
        "Evaluation profile",
        list(EVALUATION_PROFILE_LABELS.keys()),
        index=0,
    )

    evaluation_profile = EVALUATION_PROFILE_LABELS[evaluation_profile_label]

    use_clip = st.checkbox(
        "Use CLIP semantic evaluator",
        value=False,
    )

    clip_model_id = st.text_input(
        "CLIP model ID",
        value=DEFAULT_CLIP_MODEL_ID,
        disabled=not use_clip,
    )

    enable_prompt_mutation = st.checkbox(
        "Enable prompt mutation",
        value=True,
    )

    penalize_radial_artifacts = st.checkbox(
        "Penalize eye/portal-like circular artifacts",
        value=True,
    )

st.divider()

st.subheader("Recommended Test Presets")

st.write(
    """
    **Fast benchmark:** SD Turbo Fast, GPU Quick Test, 2 candidates, 4–8 steps, guidance 0.  
    **Quality benchmark:** SD 1.5 Quality, Square Cover, 2–4 candidates, 25–35 steps, guidance 7.5.
    """
)

run_button = st.button("Run Benchmark", type="primary")

if run_button:
    if not selected_model_labels:
        st.error("Please select at least one model.")
        st.stop()

    benchmark_results = []

    progress = st.progress(0)
    status = st.empty()

    total_models = len(selected_model_labels)

    for model_index, model_label in enumerate(selected_model_labels, start=1):
        model_id = available_models[model_label]

        status.write(f"Running benchmark for: {model_label}")

        config = RefinementConfig(
            prompt=prompt_bundle["prompt"],
            negative_prompt=prompt_bundle["negative_prompt"],
            model_id=model_id,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed) + model_index * 100,
            output_dir=str(OUTPUT_DIR),
            project_name=f"Benchmark-{project_name}",
            output_type=output_type,
            style=style,
            iterations=1,
            candidates_per_iteration=int(candidates_per_model),
            strength_start=0.35,
            min_strength=0.18,
            initial_image_path=None,
            reference_image_path=None,
            evaluation_profile=evaluation_profile,
            use_clip=bool(use_clip),
            clip_model_id=clip_model_id,
            penalize_radial_artifacts=bool(penalize_radial_artifacts),
            exploration_candidates_per_iteration=int(candidates_per_model),
            reference_anchor_candidates_per_iteration=0,
            artifact_escape_threshold=18.0,
            restart_when_all_candidates_have_artifacts=True,
            enable_prompt_mutation=bool(enable_prompt_mutation),
            portrait_reference_mode=False,
            keep_reference_anchor=False,
            reference_strength_start=0.22,
            reference_strength_decay=0.03,
            reference_min_strength=0.10,
        )

        with st.spinner(f"Generating candidates with {model_label}..."):
            result = run_self_refinement(config)

        benchmark_results.append(
            {
                "model_label": model_label,
                "model_id": model_id,
                "result": result,
            }
        )

        progress.progress(model_index / total_models)

    status.write("Benchmark completed.")
    st.success("Benchmark completed.")

    display_benchmark_results(benchmark_results)

    with st.expander("Raw benchmark data"):
        st.json(json.loads(json.dumps(benchmark_results, default=str)))