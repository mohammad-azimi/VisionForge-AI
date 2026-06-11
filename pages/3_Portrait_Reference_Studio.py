from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.visionforge.evaluator import DEFAULT_CLIP_MODEL_ID
from src.visionforge.presets import MODEL_PRESETS, SIZE_PRESETS
from src.visionforge.self_refiner import RefinementConfig, run_self_refinement


OUTPUT_DIR = Path("outputs")
INPUT_DIR = OUTPUT_DIR / "_portrait_reference_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Portrait Reference Studio | VisionForge-AI",
    page_icon="🎭",
    layout="wide",
)


def save_uploaded_image(uploaded_file, prefix: str) -> str | None:
    if uploaded_file is None:
        return None

    image = Image.open(uploaded_file).convert("RGB")
    safe_name = uploaded_file.name.replace(" ", "_")
    target_path = INPUT_DIR / f"{prefix}_{safe_name}"
    image.save(target_path)
    return str(target_path)


def display_history(history: list[dict]) -> None:
    st.subheader("Refinement Timeline")

    rows = []

    for item in history:
        selected = item["candidates"][0] if item.get("candidates") else {}

        rows.append(
            {
                "iteration": item["iteration"],
                "best_score": item["best_score"],
                "face_quality": selected.get("face_quality_score"),
                "face_reference_similarity": selected.get("face_reference_similarity"),
                "clip_reference_similarity": selected.get("clip_reference_similarity"),
                "reference_similarity": selected.get("reference_similarity"),
                "face_count": selected.get("face_count"),
                "selected_prompt_label": item.get("selected_prompt_label"),
                "selected_parent_source": item.get("selected_parent_source"),
                "best_image": item["best_image_path"],
            }
        )

    st.dataframe(rows, use_container_width=True)

    for item in history:
        with st.expander(f"Iteration {item['iteration']} — Best score: {item['best_score']}"):
            candidates = item.get("candidates", [])
            cols = st.columns(min(4, max(1, len(candidates))))

            for index, candidate in enumerate(candidates):
                with cols[index % len(cols)]:
                    st.image(candidate["image_path"], use_container_width=True)

                    st.caption(f"Final: {candidate.get('final_score')}")
                    st.caption(f"Face quality: {candidate.get('face_quality_score')}")
                    st.caption(f"Face reference: {candidate.get('face_reference_similarity')}")
                    st.caption(f"CLIP reference: {candidate.get('clip_reference_similarity')}")
                    st.caption(f"Face count: {candidate.get('face_count')}")
                    st.caption(f"Parent: {candidate.get('parent_source')}")
                    st.caption(f"Prompt label: {candidate.get('prompt_label')}")

                    if candidate.get("face_error"):
                        st.warning(candidate["face_error"])

                    with st.expander("Prompt used"):
                        st.write(candidate.get("prompt", ""))

                    with st.expander("Negative prompt used"):
                        st.write(candidate.get("negative_prompt", ""))


st.title("🎭 Portrait Reference Studio")
st.write(
    "A focused reference-matching workflow for portrait and face-like images. "
    "Upload a reference image, generate several candidates, score them, and refine the best one iteratively."
)

st.warning(
    "Use only images you own or have permission to use. "
    "This tool estimates visual similarity and portrait quality; it does not identify people."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Reference Image")

    uploaded_reference = st.file_uploader(
        "Upload portrait reference image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    reference_image_path = save_uploaded_image(uploaded_reference, "portrait_reference")

    if reference_image_path:
        st.image(reference_image_path, caption="Reference image", use_container_width=True)

    st.subheader("Portrait Prompt")

    prompt = st.text_area(
        "Prompt",
        value=(
            "A clean realistic portrait based on the reference image, preserving the same face structure, "
            "hairstyle, gaze direction, lighting mood, and overall identity cues, with natural skin texture, "
            "clear eyes, realistic facial proportions, high visual quality, and a clean background."
        ),
        height=150,
    )

    negative_prompt = st.text_area(
        "Negative prompt",
        value=(
            "different person, distorted face, extra face, duplicate face, extra eyes, malformed eyes, "
            "deformed mouth, bad teeth, asymmetrical face, unrealistic skin, plastic skin, blurry face, "
            "low quality, noisy, artifacts, extra limbs, text, watermark, logo"
        ),
        height=130,
    )

with right_col:
    st.subheader("Model Settings")

    selected_model = st.selectbox(
        "Model preset",
        list(MODEL_PRESETS.keys()),
        index=0,
    )

    if selected_model == "Custom model":
        model_id = st.text_input("Custom Hugging Face model ID", value="")
    else:
        model_id = MODEL_PRESETS[selected_model]

    selected_size = st.selectbox(
        "Size preset",
        list(SIZE_PRESETS.keys()),
        index=1,
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

    st.subheader("Refinement Settings")

    seed = st.number_input(
        "Base seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    iterations = st.slider(
        "Refinement iterations",
        min_value=1,
        max_value=6,
        value=3,
    )

    candidates_per_iteration = st.slider(
        "Candidates per iteration",
        min_value=2,
        max_value=6,
        value=4,
    )

    reference_anchor_candidates_per_iteration = st.slider(
        "Reference-anchor candidates per iteration",
        min_value=1,
        max_value=4,
        value=2,
    )

    exploration_candidates_per_iteration = st.slider(
        "Fresh exploration candidates per iteration",
        min_value=0,
        max_value=2,
        value=0,
    )

    num_inference_steps = st.slider(
        "Inference steps",
        min_value=4,
        max_value=50,
        value=30,
    )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=0.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
    )

    st.subheader("Strength Settings")

    strength_start = st.slider(
        "Current-parent strength",
        min_value=0.10,
        max_value=0.80,
        value=0.34,
        step=0.01,
    )

    min_strength = st.slider(
        "Minimum current-parent strength",
        min_value=0.05,
        max_value=0.50,
        value=0.16,
        step=0.01,
    )

    reference_strength_start = st.slider(
        "Reference-anchor starting strength",
        min_value=0.05,
        max_value=0.60,
        value=0.22,
        step=0.01,
    )

    reference_strength_decay = st.slider(
        "Reference-anchor strength decay",
        min_value=0.00,
        max_value=0.15,
        value=0.03,
        step=0.01,
    )

    reference_min_strength = st.slider(
        "Reference-anchor minimum strength",
        min_value=0.05,
        max_value=0.35,
        value=0.10,
        step=0.01,
    )

    st.subheader("Evaluator Settings")

    use_clip = st.checkbox(
        "Use CLIP semantic evaluator",
        value=True,
    )

    clip_model_id = st.text_input(
        "CLIP model ID",
        value=DEFAULT_CLIP_MODEL_ID,
        disabled=not use_clip,
    )

    enable_prompt_mutation = st.checkbox(
        "Enable portrait prompt mutation",
        value=True,
    )

    keep_reference_anchor = st.checkbox(
        "Keep reference anchor every iteration",
        value=True,
    )


st.divider()

run_button = st.button("Start Portrait Reference Refinement", type="primary")

if run_button:
    if not reference_image_path:
        st.error("Please upload a reference image first.")
        st.stop()

    if not model_id:
        st.error("Please select or enter a valid model ID.")
        st.stop()

    config = RefinementConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        output_dir=str(OUTPUT_DIR),
        project_name="Portrait-Reference-Studio",
        output_type="Portrait Reference Match",
        style="Realistic Portrait",
        iterations=int(iterations),
        candidates_per_iteration=int(candidates_per_iteration),
        strength_start=float(strength_start),
        min_strength=float(min_strength),
        initial_image_path=reference_image_path,
        reference_image_path=reference_image_path,
        evaluation_profile="reference_match",
        use_clip=bool(use_clip),
        clip_model_id=clip_model_id,
        penalize_radial_artifacts=False,
        exploration_candidates_per_iteration=int(exploration_candidates_per_iteration),
        reference_anchor_candidates_per_iteration=int(reference_anchor_candidates_per_iteration),
        artifact_escape_threshold=18.0,
        restart_when_all_candidates_have_artifacts=False,
        enable_prompt_mutation=bool(enable_prompt_mutation),
        portrait_reference_mode=True,
        keep_reference_anchor=bool(keep_reference_anchor),
        reference_strength_start=float(reference_strength_start),
        reference_strength_decay=float(reference_strength_decay),
        reference_min_strength=float(reference_min_strength),
    )

    with st.spinner("Running portrait reference refinement..."):
        result = run_self_refinement(config)

    st.success("Portrait reference refinement completed.")

    st.subheader("Best Output")

    if result.get("best_image_path"):
        st.image(result["best_image_path"], use_container_width=True)
        st.metric("Best score", result.get("best_score"))
        st.code(result["best_image_path"])

    display_history(result.get("history", []))

    clip_errors = []

    for item in result.get("history", []):
        for candidate in item.get("candidates", []):
            if candidate.get("clip_error"):
                clip_errors.append(candidate["clip_error"])

    if clip_errors:
        with st.expander("CLIP warnings"):
            for error in sorted(set(clip_errors)):
                st.warning(error)

    with st.expander("Raw result data"):
        st.json(json.loads(json.dumps(result, default=str)))