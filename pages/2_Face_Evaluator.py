from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.visionforge.evaluator import DEFAULT_CLIP_MODEL_ID, evaluate_image


OUTPUT_DIR = Path("outputs")
INPUT_DIR = OUTPUT_DIR / "_face_eval_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Face Evaluator | VisionForge-AI",
    page_icon="🧑",
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


st.title("🧑 Face / Portrait Evaluator")
st.write(
    "Evaluate portrait quality, face detection, and similarity to a reference image. "
    "This page is useful for testing Reference Match and portrait generation outputs."
)

st.warning(
    "Use only images you own or have permission to use. "
    "This tool estimates visual similarity and face quality; it does not identify people."
)

left_col, right_col = st.columns(2)

with left_col:
    generated_upload = st.file_uploader(
        "Generated / candidate image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    generated_path = save_uploaded_image(generated_upload, "candidate")

    if generated_path:
        st.image(generated_path, caption="Candidate image", use_container_width=True)

with right_col:
    reference_upload = st.file_uploader(
        "Reference image optional",
        type=["png", "jpg", "jpeg", "webp"],
    )

    reference_path = save_uploaded_image(reference_upload, "reference")

    if reference_path:
        st.image(reference_path, caption="Reference image", use_container_width=True)

st.divider()

prompt = st.text_area(
    "Prompt optional",
    value="A clean realistic portrait with natural facial structure and high visual quality.",
    height=100,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    evaluation_profile = st.selectbox(
        "Evaluation profile",
        ["portrait", "reference_match", "general", "portfolio"],
        index=0,
    )

with col_b:
    use_clip = st.checkbox(
        "Use CLIP semantic evaluator",
        value=False,
    )

with col_c:
    penalize_radial_artifacts = st.checkbox(
        "Penalize radial artifacts",
        value=False,
    )

clip_model_id = st.text_input(
    "CLIP model ID",
    value=DEFAULT_CLIP_MODEL_ID,
    disabled=not use_clip,
)

if st.button("Evaluate Image", type="primary"):
    if not generated_path:
        st.error("Please upload a candidate image first.")
        st.stop()

    result = evaluate_image(
        image_path=generated_path,
        reference_image_path=reference_path,
        prompt=prompt,
        evaluation_profile=evaluation_profile,
        use_clip=bool(use_clip),
        clip_model_id=clip_model_id,
        penalize_radial_artifacts=bool(penalize_radial_artifacts),
    )

    st.subheader("Main Scores")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final score", result.get("final_score"))
    m2.metric("Visual quality", result.get("visual_quality_score"))
    m3.metric("Face quality", result.get("face_quality_score"))
    m4.metric("Face reference", result.get("face_reference_similarity"))

    st.subheader("Face Detection")

    f1, f2, f3 = st.columns(3)
    f1.metric("Face count", result.get("face_count"))
    f2.metric("Reference face count", result.get("reference_face_count"))
    f3.metric("Face evaluator available", result.get("face_evaluator_available"))

    if result.get("face_error"):
        st.warning(result["face_error"])

    st.subheader("Detailed Result")
    st.json(json.loads(json.dumps(result, default=str)))
